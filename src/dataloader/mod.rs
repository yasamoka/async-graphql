//! Batch loading support, used to solve N+1 problem.
//!
//! # Examples
//!
//! ```rust
//! use async_graphql::*;
//! use async_graphql::dataloader::*;
//! use std::collections::{HashSet, HashMap};
//! use std::convert::Infallible;
//! use async_graphql::dataloader::Loader;
//!
//! /// This loader simply converts the integer key into a string value.
//! struct MyLoader;
//!
//! #[async_trait::async_trait]
//! impl Loader for MyLoader {
//!     type Key = i32;
//!     type Value = String;
//!     type Error = Infallible;
//!
//!     async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, Self::Value>, Self::Error> {
//!         // Use `MyLoader` to load data.
//!         Ok(keys.iter().copied().map(|n| (n, n.to_string())).collect())
//!     }
//! }
//!
//! struct Query;
//!
//! #[Object]
//! impl Query {
//!     async fn value(&self, ctx: &Context<'_>, n: i32) -> Option<String> {
//!         ctx.data_unchecked::<DataLoader<MyLoader>>().load_one(n).await.unwrap()
//!     }
//! }
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async move {
//! let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
//! let query = r#"
//!     {
//!         v1: value(n: 1)
//!         v2: value(n: 2)
//!         v3: value(n: 3)
//!         v4: value(n: 4)
//!         v5: value(n: 5)
//!     }
//! "#;
//! let request = Request::new(query).data(DataLoader::new(MyLoader, tokio::spawn));
//! let res = schema.execute(request).await.into_result().unwrap().data;
//!
//! assert_eq!(res, value!({
//!     "v1": "1",
//!     "v2": "2",
//!     "v3": "3",
//!     "v4": "4",
//!     "v5": "5",
//! }));
//! # });
//! ```

mod cache;

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    hash::Hash,
    sync::{Arc, Mutex},
    time::Duration,
};

pub use cache::{CacheFactory, CacheStorage, HashMapCache, LruCache, NoCache};
use futures_channel::oneshot;
use futures_timer::Delay;
use futures_util::future::BoxFuture;
use tokio::{select, sync::mpsc};
use tokio_util::sync::CancellationToken;
#[cfg(feature = "tracing")]
use tracing::{info_span, instrument, Instrument};
#[cfg(feature = "tracing")]
use tracinglib as tracing;

enum Pending<L: Loader> {
    One {
        key: L::Key,
        tx: oneshot::Sender<Result<Option<L::Value>, L::Error>>,
    },
    Many {
        keys: HashSet<L::Key>,
        use_cache_values: Option<HashMap<L::Key, L::Value>>,
        tx: oneshot::Sender<Result<HashMap<L::Key, L::Value>, L::Error>>,
    },
}

struct Requests<L: Loader> {
    keys: HashSet<L::Key>,
    pending: Vec<Pending<L>>,
    cache_storage: Box<dyn CacheStorage<Key = L::Key, Value = L::Value>>,
}

type KeysAndSender<L> = (HashSet<<L as Loader>::Key>, Vec<Pending<L>>);

impl<L: Loader> Requests<L> {
    fn new<C: CacheFactory>(cache_factory: &C) -> Self {
        Self {
            keys: Default::default(),
            pending: Vec::new(),
            cache_storage: cache_factory.create::<L::Key, L::Value>(),
        }
    }

    fn take(&mut self) -> KeysAndSender<L> {
        (
            std::mem::take(&mut self.keys),
            std::mem::take(&mut self.pending),
        )
    }
}

/// Trait for batch loading.
#[async_trait::async_trait]
pub trait Loader: Send + Sync + Clone + 'static {
    /// type of key
    type Key: Send + Sync + Hash + Eq + Clone + 'static;
    /// type of value
    type Value: Send + Sync + Clone + 'static;
    /// type of error
    type Error: Send + Clone + 'static;

    /// Load the data set specified by the `keys`.
    async fn load(
        &self,
        keys: &[Self::Key],
    ) -> Result<HashMap<Self::Key, Self::Value>, Self::Error>;
}

struct DataLoaderInner<L: Loader> {
    requests: Arc<Mutex<Requests<L>>>,
    token: CancellationToken,
    loader: L,
    delay: Duration,
    reset_delay_on_load: bool,
    max_batch_size: usize,
    disable_cache: bool,
    spawner: Box<dyn Fn(BoxFuture<'static, ()>) + Send + Sync>,
}

impl<L: Loader> DataLoaderInner<L> {
    fn new<S, R, C>(loader: L, spawner: S, cache_factory: &C) -> Self
    where
        S: Fn(BoxFuture<'static, ()>) -> R + Send + Sync + 'static,
        C: CacheFactory,
    {
        Self {
            requests: Arc::new(Mutex::new(Requests::<L>::new(cache_factory))),
            token: CancellationToken::new(),
            loader,
            delay: Duration::from_millis(1),
            reset_delay_on_load: false,
            max_batch_size: 1000,
            disable_cache: false,
            spawner: Box::new(move |fut| {
                spawner(fut);
            }),
        }
    }

    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    async fn do_load(
        loader: L,
        requests: Arc<Mutex<Requests<L>>>,
        disable_cache: bool,
        (keys, senders): KeysAndSender<L>,
    ) {
        let keys = keys.into_iter().collect::<Vec<_>>();

        match loader.load(&keys).await {
            Ok(values) => {
                // update cache
                if !disable_cache {
                    let mut requests = requests.lock().unwrap();
                    for (key, value) in &values {
                        requests
                            .cache_storage
                            .insert(Cow::Borrowed(key), Cow::Borrowed(value));
                    }
                }

                // send response
                for pending in senders {
                    match pending {
                        Pending::One { key, tx } => {
                            let value = values.get(&key).map(|value| value.clone());
                            tx.send(Ok(value)).ok();
                        }
                        Pending::Many {
                            keys,
                            use_cache_values,
                            tx,
                        } => {
                            let mut res = HashMap::new();
                            if let Some(use_cache_values) = use_cache_values {
                                res.extend(use_cache_values);
                            }

                            for key in &keys {
                                res.extend(
                                    values.get(key).map(|value| (key.clone(), value.clone())),
                                );
                            }

                            tx.send(Ok(res)).ok();
                        }
                    }
                }
            }
            Err(err) => {
                for pending in senders {
                    match pending {
                        Pending::One { tx, .. } => {
                            tx.send(Err(err.clone())).ok();
                        }
                        Pending::Many { tx, .. } => {
                            tx.send(Err(err.clone())).ok();
                        }
                    }
                }
            }
        }
    }

    fn load_one(&mut self, key: L::Key, tx: oneshot::Sender<Result<Option<L::Value>, L::Error>>) {
        let action = {
            let mut requests = self.requests.lock().unwrap();
            let prev_count = requests.keys.len();

            if !self.disable_cache {
                if let Some(value) = requests.cache_storage.get(&key) {
                    let value = Some(value.clone());
                    tx.send(Ok(value)).ok();
                    return;
                }
            };

            requests.keys.insert(key.clone());
            requests.pending.push(Pending::One {
                key: key.clone(),
                tx,
            });

            self.get_action(&mut requests, prev_count)
        };

        self.execute_action(action);
    }

    fn load_many<I>(
        &mut self,
        keys: I,
        tx: oneshot::Sender<Result<HashMap<L::Key, L::Value>, L::Error>>,
    ) where
        I: IntoIterator<Item = L::Key>,
    {
        let action = {
            let mut requests = self.requests.lock().unwrap();
            let prev_count = requests.keys.len();
            let mut keys_set = HashSet::new();

            let use_cache_values = if self.disable_cache {
                keys_set = keys.into_iter().collect();

                if keys_set.is_empty() {
                    tx.send(Ok(Default::default())).ok();
                    return;
                }

                None
            } else {
                let mut use_cache_values = HashMap::new();
                for key in keys {
                    if let Some(value) = requests.cache_storage.get(&key) {
                        // Already in cache
                        use_cache_values.insert(key.clone(), value.clone());
                    } else {
                        keys_set.insert(key);
                    }
                }

                if keys_set.is_empty() {
                    tx.send(Ok(if use_cache_values.is_empty() {
                        Default::default()
                    } else {
                        use_cache_values
                    }))
                    .ok();
                    return;
                }

                Some(use_cache_values)
            };

            requests.keys.extend(keys_set.clone());
            requests.pending.push(Pending::Many {
                keys: keys_set,
                use_cache_values,
                tx,
            });

            self.get_action(&mut requests, prev_count)
        };

        self.execute_action(action);
    }

    fn get_action(&self, requests: &mut Requests<L>, prev_count: usize) -> Action<L> {
        if requests.keys.len() >= self.max_batch_size {
            Action::ImmediateLoad(requests.take())
        } else {
            if self.reset_delay_on_load {
                if requests.keys.is_empty() {
                    Action::Delay
                } else {
                    Action::RestartFetch(prev_count > 0)
                }
            } else {
                if !requests.keys.is_empty() && prev_count == 0 {
                    Action::StartFetch
                } else {
                    Action::Delay
                }
            }
        }
    }

    fn execute_action(&mut self, action: Action<L>) {
        match action {
            Action::ImmediateLoad(keys) => {
                let loader = self.loader.clone();
                let requests = self.requests.clone();
                let disable_cache = self.disable_cache;
                let task =
                    async move { Self::do_load(loader, requests, disable_cache, keys).await };
                #[cfg(feature = "tracing")]
                let task = task
                    .instrument(info_span!("immediate_load"))
                    .in_current_span();

                (self.spawner)(Box::pin(task));
            }
            Action::StartFetch => {
                let loader = self.loader.clone();
                let requests = self.requests.clone();
                let disable_cache = self.disable_cache;
                let delay = self.delay;

                let task = async move {
                    Delay::new(delay).await;

                    let keys = {
                        let mut requests = requests.lock().unwrap();
                        requests.take()
                    };

                    if !keys.0.is_empty() {
                        Self::do_load(loader, requests, disable_cache, keys).await
                    }
                };
                #[cfg(feature = "tracing")]
                let task = task.instrument(info_span!("start_fetch")).in_current_span();
                (self.spawner)(Box::pin(task))
            }
            Action::RestartFetch(restart) => {
                let loader = self.loader.clone();
                let requests = self.requests.clone();
                let disable_cache = self.disable_cache;
                let delay = self.delay;

                let child_token = {
                    if restart {
                        self.token.cancel();
                        self.token = CancellationToken::new();
                    }
                    self.token.child_token()
                };

                let task = async move {
                    select! {
                        _ = child_token.cancelled() => {}
                        _ = Delay::new(delay) => {
                            let keys = {
                                let mut requests = requests.lock().unwrap();
                                requests.take()
                            };

                            if !keys.0.is_empty() {
                                Self::do_load(loader, requests, disable_cache, keys).await
                            }
                        }
                    }
                };
                #[cfg(feature = "tracing")]
                let task = task
                    .instrument(info_span!("restart_fetch"))
                    .in_current_span();
                (self.spawner)(Box::pin(task))
            }
            Action::Delay => {}
        }
    }

    fn feed_many<I>(&self, items: I)
    where
        I: IntoIterator<Item = (L::Key, L::Value)>,
    {
        let mut requests = self.requests.lock().unwrap();

        for (key, value) in items {
            requests
                .cache_storage
                .insert(Cow::Owned(key), Cow::Owned(value));
        }
    }

    fn feed_one(&self, key: L::Key, value: L::Value) {
        self.feed_many(std::iter::once((key, value)));
    }

    fn clear(&self) {
        let mut requests = self.requests.lock().unwrap();
        requests.cache_storage.clear();
    }

    fn get_cached_values(&mut self) -> HashMap<L::Key, L::Value> {
        let requests = self.requests.lock().unwrap();
        requests
            .cache_storage
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Data loader.
///
/// Reference: <https://github.com/facebook/dataloader>
pub struct DataLoader<L: Loader> {
    tx: mpsc::Sender<Request<L>>,
}

enum Action<L: Loader> {
    ImmediateLoad(KeysAndSender<L>),
    StartFetch,
    RestartFetch(bool),
    Delay,
}

impl<L: Loader> DataLoader<L> {
    /// Use 'Loader' to create a builder for a [DataLoader] that does not cache records.
    pub fn build<S, R>(loader: L, spawner: S) -> DataLoaderBuilder<L>
    where
        S: Fn(BoxFuture<'static, ()>) -> R + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel(1000000);
        DataLoaderBuilder {
            dataloader: Self { tx },
            inner: DataLoaderInner::new(loader, spawner, &NoCache),
            rx,
        }
    }
}

enum Request<L: Loader> {
    LoadOne {
        key: L::Key,
        tx: oneshot::Sender<Result<Option<L::Value>, L::Error>>,
    },
    LoadMany {
        keys: Vec<L::Key>,
        tx: oneshot::Sender<Result<HashMap<L::Key, L::Value>, L::Error>>,
    },
    FeedOne {
        key: L::Key,
        value: L::Value,
        tx: oneshot::Sender<()>,
    },
    FeedMany {
        items: Vec<(L::Key, L::Value)>,
        tx: oneshot::Sender<()>,
    },
    Clear {
        tx: oneshot::Sender<()>,
    },
    GetCachedValues {
        tx: oneshot::Sender<HashMap<L::Key, L::Value>>,
    },
}

async fn dataloader_task<L: Loader>(
    mut inner: DataLoaderInner<L>,
    mut rx: mpsc::Receiver<Request<L>>,
) {
    while let Some(request) = rx.recv().await {
        match request {
            Request::LoadOne { key, tx } => {
                inner.load_one(key, tx);
            }
            Request::LoadMany { keys, tx } => {
                inner.load_many(keys, tx);
            }
            Request::FeedOne { key, value, tx } => {
                let v = inner.feed_one(key, value);
                tx.send(v).ok();
            }
            Request::FeedMany { items, tx } => {
                let v = inner.feed_many(items);
                tx.send(v).ok();
            }
            Request::Clear { tx } => {
                inner.clear();
                tx.send(()).ok();
            }
            Request::GetCachedValues { tx } => {
                let v = inner.get_cached_values();
                tx.send(v).ok();
            }
        }
    }
}

/// Data loader builder
pub struct DataLoaderBuilder<L: Loader> {
    dataloader: DataLoader<L>,
    inner: DataLoaderInner<L>,
    rx: mpsc::Receiver<Request<L>>,
}

impl<L: Loader> DataLoaderBuilder<L> {
    /// Specify the delay time for loading data, the default is `1ms`.
    #[must_use]
    pub fn delay(self, delay: Duration) -> Self {
        let mut inner = self.inner;
        inner.delay = delay;
        Self { inner, ..self }
    }

    /// Specify whether to reset the delay on load, the default is `false`
    #[must_use]
    pub fn reset_delay_on_load(self, reset_delay_on_load: bool) -> Self {
        let mut inner = self.inner;
        inner.reset_delay_on_load = reset_delay_on_load;
        Self { inner, ..self }
    }

    /// pub fn Specify the max batch size for loading data, the default is
    /// `1000`.
    ///
    /// If the keys waiting to be loaded reach the threshold, they are loaded
    /// immediately.
    #[must_use]
    pub fn max_batch_size(self, max_batch_size: usize) -> Self {
        let mut inner = self.inner;
        inner.max_batch_size = max_batch_size;
        Self { inner, ..self }
    }

    /// Finish building the [DataLoader].
    pub fn finish(self) -> DataLoader<L> {
        tokio::spawn(dataloader_task(self.inner, self.rx));
        self.dataloader
    }
}

impl<L: Loader> DataLoader<L> {
    /// Use 'Loader' to create a builder for a [DataLoader] with a cache factory.
    pub fn build_with_cache<S, R, C>(
        loader: L,
        spawner: S,
        cache_factory: C,
    ) -> DataLoaderBuilder<L>
    where
        S: Fn(BoxFuture<'static, ()>) -> R + Send + Sync + 'static,
        C: CacheFactory,
    {
        let (tx, rx) = mpsc::channel(1000000);
        DataLoaderBuilder {
            dataloader: Self { tx },
            inner: DataLoaderInner::new(loader, spawner, &cache_factory),
            rx,
        }
    }

    /// Use this `DataLoader` load a data.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn load_one(&self, key: L::Key) -> Result<Option<L::Value>, L::Error> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .clone()
            .send(Request::LoadOne {
                key: key.clone(),
                tx,
            })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    /// Use this `DataLoader` to load some data.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn load_many<I>(&self, keys: I) -> Result<HashMap<L::Key, L::Value>, L::Error>
    where
        I: IntoIterator<Item = L::Key>,
    {
        let (tx, rx) = oneshot::channel();
        self.tx
            .clone()
            .send(Request::LoadMany {
                keys: keys.into_iter().collect(),
                tx,
            })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    /// Feed some data into the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn feed_one(&self, key: L::Key, value: L::Value) {
        let (tx, rx) = oneshot::channel();
        self.tx
            .clone()
            .send(Request::FeedOne { key, value, tx })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    /// Feed some data into the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn feed_many<I>(&self, items: I)
    where
        I: IntoIterator<Item = (L::Key, L::Value)>,
    {
        let (tx, rx) = oneshot::channel();
        self.tx
            .clone()
            .send(Request::FeedMany {
                items: items.into_iter().collect(),
                tx,
            })
            .await
            .unwrap();
        rx.await.unwrap()
    }

    /// Clears the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn clear(&self) {
        let (tx, rx) = oneshot::channel();
        self.tx.clone().send(Request::Clear { tx }).await.unwrap();
        rx.await.unwrap()
    }

    /// Gets all values in the cache.
    pub async fn get_cached_values(&self) -> HashMap<L::Key, L::Value> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .clone()
            .send(Request::GetCachedValues { tx })
            .await
            .unwrap();
        rx.await.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fnv::FnvBuildHasher;

    use super::*;

    #[derive(Clone)]
    struct MyLoader;

    #[async_trait::async_trait]
    impl Loader for MyLoader {
        type Key = i32;
        type Value = i32;
        type Error = ();

        async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, i32>, ()> {
            assert!(keys.len() <= 10);
            Ok(keys.iter().copied().map(|k| (k, k)).collect())
        }
    }

    #[derive(Clone)]
    struct MyLoader64;

    #[async_trait::async_trait]
    impl Loader for MyLoader64 {
        type Key = i64;
        type Value = i64;
        type Error = ();

        async fn load(&self, keys: &[i64]) -> Result<HashMap<i64, i64>, ()> {
            assert!(keys.len() <= 10);
            Ok(keys.iter().copied().map(|k| (k, k)).collect())
        }
    }

    #[tokio::test]
    async fn test_dataloader() {
        let loader = Arc::new(
            DataLoader::build(MyLoader, tokio::spawn)
                .max_batch_size(10)
                .finish(),
        );
        assert_eq!(
            futures_util::future::try_join_all((0..100i32).map({
                let loader = loader.clone();
                move |n| {
                    let loader = loader.clone();
                    async move { loader.load_one(n).await }
                }
            }))
            .await
            .unwrap(),
            (0..100).map(Option::Some).collect::<Vec<_>>()
        );

        let loader = Arc::new(
            DataLoader::build(MyLoader64, tokio::spawn)
                .max_batch_size(10)
                .finish(),
        );
        assert_eq!(
            futures_util::future::try_join_all((0..100i64).map({
                let loader = loader.clone();
                move |n| {
                    let loader = loader.clone();
                    async move { loader.load_one(n).await }
                }
            }))
            .await
            .unwrap(),
            (0..100).map(Option::Some).collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_duplicate_keys() {
        let loader = Arc::new(
            DataLoader::build(MyLoader, tokio::spawn)
                .max_batch_size(10)
                .finish(),
        );
        assert_eq!(
            futures_util::future::try_join_all([1, 3, 5, 1, 7, 8, 3, 7].iter().copied().map({
                let loader = loader.clone();
                move |n| {
                    let loader = loader.clone();
                    async move { loader.load_one(n).await }
                }
            }))
            .await
            .unwrap(),
            [1, 3, 5, 1, 7, 8, 3, 7]
                .iter()
                .copied()
                .map(Option::Some)
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn test_dataloader_load_empty() {
        let loader = DataLoader::build(MyLoader, tokio::spawn).finish();
        assert!(loader.load_many(vec![]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_dataloader_with_cache() {
        let loader =
            DataLoader::build_with_cache(MyLoader, tokio::spawn, HashMapCache::default()).finish();
        loader.feed_many(vec![(1, 10), (2, 20), (3, 30)]).await;

        // All from the cache
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 10), (2, 20), (3, 30)].into_iter().collect()
        );

        // Part from the cache
        assert_eq!(
            loader.load_many(vec![1, 5, 6]).await.unwrap(),
            vec![(1, 10), (5, 5), (6, 6)].into_iter().collect()
        );

        // All from the loader
        assert_eq!(
            loader.load_many(vec![8, 9, 10]).await.unwrap(),
            vec![(8, 8), (9, 9), (10, 10)].into_iter().collect()
        );

        // Clear cache
        loader.clear().await;
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_with_cache_hashmap_fnv() {
        let loader = DataLoader::build_with_cache(
            MyLoader,
            tokio::spawn,
            HashMapCache::<FnvBuildHasher>::new(),
        )
        .finish();
        loader.feed_many(vec![(1, 10), (2, 20), (3, 30)]).await;

        // All from the cache
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 10), (2, 20), (3, 30)].into_iter().collect()
        );

        // Part from the cache
        assert_eq!(
            loader.load_many(vec![1, 5, 6]).await.unwrap(),
            vec![(1, 10), (5, 5), (6, 6)].into_iter().collect()
        );

        // All from the loader
        assert_eq!(
            loader.load_many(vec![8, 9, 10]).await.unwrap(),
            vec![(8, 8), (9, 9), (10, 10)].into_iter().collect()
        );

        // Clear cache
        loader.clear().await;
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_dead_lock() {
        #[derive(Clone)]
        struct MyDelayLoader;

        #[async_trait::async_trait]
        impl Loader for MyDelayLoader {
            type Key = i32;
            type Value = i32;
            type Error = ();

            async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, i32>, ()> {
                tokio::time::sleep(Duration::from_secs(1)).await;
                Ok(keys.iter().copied().map(|k| (k, k)).collect())
            }
        }

        let loader = Arc::new(
            DataLoader::build_with_cache(MyDelayLoader, tokio::spawn, NoCache)
                .delay(Duration::from_secs(1))
                .finish(),
        );
        let handle = tokio::spawn({
            let loader = loader.clone();
            async move {
                loader.load_many(vec![1, 2, 3]).await.unwrap();
            }
        });

        tokio::time::sleep(Duration::from_millis(500)).await;
        handle.abort();
        loader.load_many(vec![4, 5, 6]).await.unwrap();
    }
}

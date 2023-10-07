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
//! impl Loader<i32> for MyLoader {
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
    ops::DerefMut,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
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

#[allow(clippy::type_complexity)]
struct ResSender<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> {
    use_cache_values: Option<HashMap<K, T::Value>>,
    tx: oneshot::Sender<Result<HashMap<K, T::Value>, T::Error>>,
}

enum KeySet<K: Send + Sync + Hash + Eq + Clone + 'static> {
    One(K),
    Many(HashSet<K>),
}

struct Requests<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> {
    keys: HashSet<K>,
    pending: Vec<(KeySet<K>, ResSender<K, T>)>,
    cache_storage: Box<dyn CacheStorage<Key = K, Value = T::Value>>,
    disable_cache: bool,
}

type KeysAndSender<K, T> = (HashSet<K>, Vec<(KeySet<K>, ResSender<K, T>)>);

impl<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> Requests<K, T> {
    fn new<C: CacheFactory>(cache_factory: &C) -> Self {
        Self {
            keys: Default::default(),
            pending: Vec::new(),
            cache_storage: cache_factory.create::<K, T::Value>(),
            disable_cache: false,
        }
    }

    fn take(&mut self) -> KeysAndSender<K, T> {
        (
            std::mem::take(&mut self.keys),
            std::mem::take(&mut self.pending),
        )
    }
}

/// Trait for batch loading.
#[async_trait::async_trait]
pub trait Loader<K: Send + Sync + Hash + Eq + Clone + 'static>: Send + Sync + 'static {
    /// type of value.
    type Value: Send + Sync + Clone + 'static;

    /// Type of error.
    type Error: Send + Clone + 'static;

    /// Load the data set specified by the `keys`.
    async fn load(&self, keys: &[K]) -> Result<HashMap<K, Self::Value>, Self::Error>;
}

struct DataLoaderInner<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> {
    requests: Mutex<Requests<K, T>>,
    token: Mutex<CancellationToken>,
    loader: T,
}

impl<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> DataLoaderInner<K, T> {
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    async fn do_load(&self, disable_cache: bool, (keys, senders): KeysAndSender<K, T>) {
        let keys = keys.into_iter().collect::<Vec<_>>();

        match self.loader.load(&keys).await {
            Ok(values) => {
                // update cache
                let mut requests = self.requests.lock().unwrap();
                let disable_cache = requests.disable_cache || disable_cache;
                if !disable_cache {
                    for (key, value) in &values {
                        requests
                            .cache_storage
                            .insert(Cow::Borrowed(key), Cow::Borrowed(value));
                    }
                }

                // send response
                for (keys, sender) in senders {
                    let mut res = HashMap::new();
                    if let Some(use_cache_values) = sender.use_cache_values {
                        res.extend(use_cache_values);
                    }
                    match keys {
                        KeySet::Many(keys) => {
                            for key in &keys {
                                res.extend(
                                    values.get(key).map(|value| (key.clone(), value.clone())),
                                );
                            }
                        }
                        KeySet::One(key) => {
                            res.extend(values.get(&key).map(|value| (key.clone(), value.clone())));
                        }
                    }
                    sender.tx.send(Ok(res)).ok();
                }
            }
            Err(err) => {
                for (_, sender) in senders {
                    sender.tx.send(Err(err.clone())).ok();
                }
            }
        }
    }
}

/// Data loader.
///
/// Reference: <https://github.com/facebook/dataloader>
pub struct DataLoader<T: Loader<K>, K: Send + Sync + Hash + Eq + Clone + 'static, C = NoCache> {
    inner: Arc<DataLoaderInner<K, T>>,
    tx: mpsc::Sender<LoadKey<K>>,
    cache_factory: C,
    delay: Duration,
    reset_delay_on_load: bool,
    max_batch_size: usize,
    disable_cache: AtomicBool,
    spawner: Box<dyn Fn(BoxFuture<'static, ()>) + Send + Sync>,
}

enum LoadKey<K: Send + Sync + Hash + Eq + Clone + 'static> {
    One(K),
    Many(Vec<K>),
}

enum Action<K: Send + Sync + Hash + Eq + Clone + 'static, T: Loader<K>> {
    ImmediateLoad(KeysAndSender<K, T>),
    StartFetch,
    RestartFetch(bool),
    Delay,
}

impl<T: Loader<K>, K: Send + Sync + Hash + Eq + Clone + 'static> DataLoader<T, K, NoCache> {
    /// Use `Loader` to create a [DataLoader] that does not cache records.
    pub fn new<S, R>(loader: T, spawner: S) -> Self
    where
        S: Fn(BoxFuture<'static, ()>) -> R + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel(1000000);

        Self {
            inner: Arc::new(DataLoaderInner {
                requests: Mutex::new(Requests::<K, T>::new(&NoCache)),
                token: Mutex::new(CancellationToken::new()),
                loader,
            }),
            tx,
            cache_factory: NoCache,
            delay: Duration::from_millis(1),
            reset_delay_on_load: false,
            max_batch_size: 1000,
            disable_cache: false.into(),
            spawner: Box::new(move |fut| {
                spawner(fut);
            }),
        }
    }
}

impl<T: Loader<K>, K: Send + Sync + Hash + Eq + Clone + 'static, C: CacheFactory>
    DataLoader<T, K, C>
{
    /// Use `Loader` to create a [DataLoader] with a cache factory.
    pub fn with_cache<S, R>(loader: T, spawner: S, cache_factory: C) -> Self
    where
        S: Fn(BoxFuture<'static, ()>) -> R + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel(1000000);

        Self {
            inner: Arc::new(DataLoaderInner {
                requests: Mutex::new(Requests::<K, T>::new(&cache_factory)),
                token: Mutex::new(CancellationToken::new()),
                loader,
            }),
            tx,
            cache_factory,
            delay: Duration::from_millis(1),
            reset_delay_on_load: false,
            max_batch_size: 1000,
            disable_cache: false.into(),
            spawner: Box::new(move |fut| {
                spawner(fut);
            }),
        }
    }

    /// Specify the delay time for loading data, the default is `1ms`.
    #[must_use]
    pub fn delay(self, delay: Duration) -> Self {
        Self { delay, ..self }
    }

    /// Specify whether to reset the delay on load, the default is `false`
    #[must_use]
    pub fn reset_delay_on_load(self, reset_delay_on_load: bool) -> Self {
        Self {
            reset_delay_on_load,
            ..self
        }
    }

    /// pub fn Specify the max batch size for loading data, the default is
    /// `1000`.
    ///
    /// If the keys waiting to be loaded reach the threshold, they are loaded
    /// immediately.
    #[must_use]
    pub fn max_batch_size(self, max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            ..self
        }
    }

    /// Get the loader.
    #[inline]
    pub fn loader(&self) -> &T {
        &self.inner.loader
    }

    /// Enable/Disable cache of all loaders.
    pub fn enable_all_cache(&self, enable: bool) {
        self.disable_cache.store(!enable, Ordering::SeqCst);
    }

    /// Enable/Disable cache of specified loader.
    pub fn enable_cache(&self, enable: bool) {
        let mut requests = self.inner.requests.lock().unwrap();
        requests.disable_cache = !enable;
    }

    /// Use this `DataLoader` load a data.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn load_one(&self, key: K) -> Result<Option<T::Value>, T::Error> {
        let (action, rx) = {
            let mut guard = self.inner.requests.lock().unwrap();
            let requests = &mut *guard;
            let prev_count = requests.keys.len();

            let use_cache_values =
                if requests.disable_cache || self.disable_cache.load(Ordering::SeqCst) {
                    None
                } else {
                    let mut use_cache_values = HashMap::new();
                    if let Some(value) = requests.cache_storage.get(&key) {
                        use_cache_values.insert(key.clone(), value.clone());
                        return Ok(Some(value.clone()));
                    }

                    Some(use_cache_values)
                };

            requests.keys.insert(key.clone());
            let (tx, rx) = oneshot::channel();
            requests.pending.push((
                KeySet::One(key.clone()),
                ResSender {
                    use_cache_values,
                    tx,
                },
            ));

            (self.get_action(requests, prev_count), rx)
        };

        self.execute_action(action);

        let mut values = rx.await.unwrap()?;
        Ok(values.remove(&key))
    }

    /// Use this `DataLoader` to load some data.
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn load_many<I>(&self, keys: I) -> Result<HashMap<K, T::Value>, T::Error>
    where
        I: IntoIterator<Item = K>,
    {
        let (action, rx) = {
            let mut guard = self.inner.requests.lock().unwrap();
            let requests = &mut *guard;
            let prev_count = requests.keys.len();
            let mut keys_set = HashSet::new();

            let use_cache_values =
                if requests.disable_cache || self.disable_cache.load(Ordering::SeqCst) {
                    keys_set = keys.into_iter().collect();

                    if keys_set.is_empty() {
                        return Ok(Default::default());
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
                        return Ok(if use_cache_values.is_empty() {
                            Default::default()
                        } else {
                            use_cache_values
                        });
                    }

                    Some(use_cache_values)
                };

            requests.keys.extend(keys_set.clone());
            let (tx, rx) = oneshot::channel();
            requests.pending.push((
                KeySet::Many(keys_set),
                ResSender {
                    use_cache_values,
                    tx,
                },
            ));

            (self.get_action(requests, prev_count), rx)
        };

        self.execute_action(action);

        rx.await.unwrap()
    }

    fn get_action(&self, requests: &mut Requests<K, T>, prev_count: usize) -> Action<K, T> {
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

    fn execute_action(&self, action: Action<K, T>) {
        match action {
            Action::ImmediateLoad(keys) => {
                let inner = self.inner.clone();
                let disable_cache = self.disable_cache.load(Ordering::SeqCst);
                let task = async move { inner.do_load(disable_cache, keys).await };
                #[cfg(feature = "tracing")]
                let task = task
                    .instrument(info_span!("immediate_load"))
                    .in_current_span();

                (self.spawner)(Box::pin(task));
            }
            Action::StartFetch => {
                let inner = self.inner.clone();
                let disable_cache = self.disable_cache.load(Ordering::SeqCst);
                let delay = self.delay;

                let task = async move {
                    Delay::new(delay).await;

                    let keys = {
                        let mut requests = inner.requests.lock().unwrap();
                        requests.take()
                    };

                    if !keys.0.is_empty() {
                        inner.do_load(disable_cache, keys).await
                    }
                };
                #[cfg(feature = "tracing")]
                let task = task.instrument(info_span!("start_fetch")).in_current_span();
                (self.spawner)(Box::pin(task))
            }
            Action::RestartFetch(restart) => {
                let inner = self.inner.clone();
                let disable_cache = self.disable_cache.load(Ordering::SeqCst);
                let delay = self.delay;

                let child_token = {
                    let mut guard = self.inner.token.lock().unwrap();
                    let token = guard.deref_mut();
                    if restart {
                        token.cancel();
                        *token = CancellationToken::new();
                    }
                    token.child_token()
                };

                let task = async move {
                    select! {
                        _ = child_token.cancelled() => {}
                        _ = Delay::new(delay) => {
                            let keys = {
                                let mut requests = inner.requests.lock().unwrap();
                                requests.take()
                            };

                            if !keys.0.is_empty() {
                                inner.do_load(disable_cache, keys).await
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

    /// Feed some data into the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn feed_many<I>(&self, values: I)
    where
        I: IntoIterator<Item = (K, T::Value)>,
    {
        let mut guard = self.inner.requests.lock().unwrap();
        let requests = &mut *guard;
        for (key, value) in values {
            requests
                .cache_storage
                .insert(Cow::Owned(key), Cow::Owned(value));
        }
    }

    /// Feed some data into the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub async fn feed_one(&self, key: K, value: T::Value) {
        self.feed_many(std::iter::once((key, value))).await;
    }

    /// Clears the cache.
    ///
    /// **NOTE: If the cache type is [NoCache], this function will not take
    /// effect. **
    #[cfg_attr(feature = "tracing", instrument(skip_all))]
    pub fn clear(&self) {
        let mut requests = self.inner.requests.lock().unwrap();
        requests.cache_storage.clear();
    }

    /// Gets all values in the cache.
    pub fn get_cached_values(&self) -> HashMap<K, T::Value> {
        let requests = self.inner.requests.lock().unwrap();
        requests
            .cache_storage
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fnv::FnvBuildHasher;

    use super::*;

    struct MyLoader;

    #[async_trait::async_trait]
    impl Loader<i32> for MyLoader {
        type Value = i32;
        type Error = ();

        async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, Self::Value>, Self::Error> {
            assert!(keys.len() <= 10);
            Ok(keys.iter().copied().map(|k| (k, k)).collect())
        }
    }

    #[async_trait::async_trait]
    impl Loader<i64> for MyLoader {
        type Value = i64;
        type Error = ();

        async fn load(&self, keys: &[i64]) -> Result<HashMap<i64, Self::Value>, Self::Error> {
            assert!(keys.len() <= 10);
            Ok(keys.iter().copied().map(|k| (k, k)).collect())
        }
    }

    #[tokio::test]
    async fn test_dataloader() {
        let loader = Arc::new(DataLoader::new(MyLoader, tokio::spawn).max_batch_size(10));
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

        let loader = Arc::new(DataLoader::new(MyLoader, tokio::spawn).max_batch_size(10));
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
        let loader = Arc::new(DataLoader::new(MyLoader, tokio::spawn).max_batch_size(10));
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
        let loader = DataLoader::<_, i32>::new(MyLoader, tokio::spawn);
        assert!(loader.load_many(vec![]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_dataloader_with_cache() {
        let loader = DataLoader::with_cache(MyLoader, tokio::spawn, HashMapCache::default());
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
        loader.clear();
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_with_cache_hashmap_fnv() {
        let loader = DataLoader::with_cache(
            MyLoader,
            tokio::spawn,
            HashMapCache::<FnvBuildHasher>::new(),
        );
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
        loader.clear();
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_disable_all_cache() {
        let loader = DataLoader::with_cache(MyLoader, tokio::spawn, HashMapCache::default());
        loader.feed_many(vec![(1, 10), (2, 20), (3, 30)]).await;

        // All from the loader
        loader.enable_all_cache(false);
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );

        // All from the cache
        loader.enable_all_cache(true);
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 10), (2, 20), (3, 30)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_disable_cache() {
        let loader = DataLoader::with_cache(MyLoader, tokio::spawn, HashMapCache::default());
        loader.feed_many(vec![(1, 10), (2, 20), (3, 30)]).await;

        // All from the loader
        loader.enable_cache(false);
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 1), (2, 2), (3, 3)].into_iter().collect()
        );

        // All from the cache
        loader.enable_cache(true);
        assert_eq!(
            loader.load_many(vec![1, 2, 3]).await.unwrap(),
            vec![(1, 10), (2, 20), (3, 30)].into_iter().collect()
        );
    }

    #[tokio::test]
    async fn test_dataloader_dead_lock() {
        struct MyDelayLoader;

        #[async_trait::async_trait]
        impl Loader<i32> for MyDelayLoader {
            type Value = i32;
            type Error = ();

            async fn load(&self, keys: &[i32]) -> Result<HashMap<i32, Self::Value>, Self::Error> {
                tokio::time::sleep(Duration::from_secs(1)).await;
                Ok(keys.iter().copied().map(|k| (k, k)).collect())
            }
        }

        let loader = Arc::new(
            DataLoader::with_cache(MyDelayLoader, tokio::spawn, NoCache)
                .delay(Duration::from_secs(1)),
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

fn test_drop_on_event_handler() {
    // When the handler receives a system event, it notifies the
    // service that holds the associated resource. If this notification results in
    // the service being dropped, the resource will also be dropped.
    //
    // Previously, there was a deadlock scenario where the handler, while
    // notifying, held a lock and the service being dropped attempted to acquire
    // that same lock in order to clean up state.
    //
    // To simulate this case, we create a fake executor that does nothing when
    // the service is notified. This simulates an executor in the process of
    // shutting down. Then, when the service handle is dropped, the service itself is
    // dropped.

    let handler = runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let (event_tx, event_rx) = mpsc::channel();

    // Define a service that just processes events
    let service = Arc::new(Service::new(async move {
        loop {
            let event = event_rx.recv().await.unwrap();
            // Process the event
            handle_event(event);
        }
    }));

    {
        let _enter = handler.enter();
        let waker = waker_ref(&service);
        let mut cx = Context::from_waker(&waker);
        assert_pending!(service.future.lock().unwrap().as_mut().poll(&mut cx));
    }

    // Get the event
    let event = event_rx.recv().await.unwrap();

    drop(service);

    // Establish a connection to the service
    connect_service(event);

    // Force the handler to turn
    handler.block_on(async {});
}

struct Service {
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Service {
    fn new(future: Pin<Box<dyn Future<Output = ()> + Send>>) -> Self {
        Service { future: Mutex::new(future) }
    }
}

fn waker_ref(service: &Arc<Service>) -> Arc<Task> {
    Arc::clone(&service)
}

trait Context {
    fn from_waker(waker: &Arc<Task>) -> Self;
}

struct Task;

impl Context for Task {
    fn from_waker(waker: &Arc<Task>) -> Self {
        *waker
    }
}

async fn handle_event(event: Event) {}

type Event = ();

fn connect_service(event: Event) {}

fn generate_struct_for_pred() {
    check(
        r#"
struct Bar<'lt, T, const C: usize> where for<'b> $0 {}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Bar<…> Bar<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

fn is_debug_serversession() {
        use std::{println, Vec};
        let protocol_version = ProtocolVersion::TLSv1_3;
        let cipher_suite = CipherSuite::TLS13_AES_128_GCM_SHA256;
        let session_id = vec![1, 2, 3];
        let unix_time = UnixTime::now();
        let opaque_value = 0x12345678;
        let ssv = ServerSessionValue::new(
            None,
            protocol_version,
            cipher_suite,
            &session_id,
            None,
            None,
            vec![4, 5, 6],
            unix_time,
            opaque_value,
        );
        println!("{:?}", ssv);
    }

fn transform_data_block_body() {
        check_assist(
            move_const_to_impl,
            r#"
struct T;
impl T {
    fn process() -> i32 {
        /// method comment
        const D$0: i32 = {
            let x = 5;
            let y = 6;
            x * y
        };

        D * D
    }
}
"#,
            r#"
struct T;
impl T {
    /// method comment
    const D: i32 = {
        let x = 5;
        let y = 6;
        x * y
    };

    fn process() -> i32 {
        Self::D * Self::D
    }
}
"#,
        );
    }


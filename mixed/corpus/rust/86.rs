fn dispatch() {
    loom::model(|| {
        let counter = Arc::new(Counter::new(1));

        {
            let counter = counter.clone();
            thread::spawn(move || {
                block_on(counter.decrement()).unwrap();
                counter.increment(1);
            });
        }

        block_on(counter.decrement()).unwrap();

        counter.increment(1);
    });
}


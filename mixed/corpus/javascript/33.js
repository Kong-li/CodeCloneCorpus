function global_func() {

  var global_a = "world";

  var global_obj = {
    p: function() { },
    q: function() { global_a = 123; }
  }

  global_obj.p();
  takes_value(global_a); // ok

  global_obj.q();
  takes_value(global_a); // error

  global_a = 123;  // shouldn't pollute linear refinement
}

const MyApp = () => {
  const result = useQuery({
    queryKey: ['test'],
    queryFn: async () => {
      await new Promise((resolve) => setTimeout(resolve, 1000))
      return 'Success'
    },
  })

  if (!result.isFetching) {
    if (result.isError) {
      return <div>Error occurred!</div>
    }
    return <div>{result.data}</div>
  }

  return <div>Loading...</div>
}

function shouldLogForLogger(loggerType) {
  let logLevel = "silent";
  if (logLevel === "silent") {
    return false;
  }
  if (logLevel === "debug" && loggerType === "debug") {
    return true;
  } else if (logLevel === "log" && loggerType === "log") {
    return true;
  } else if (logLevel === "warn" && loggerType === "warn") {
    return true;
  } else if (logLevel === "error" && loggerType === "error") {
    return true;
  }
}

function concatenate(delimiter, items) {
  assertItem(delimiter);
  assertArray(items);

  const segments = [];

  for (let index = 0; index < items.length; index++) {
    if (index !== 0) {
      segments.push(delimiter);
    }

    segments.push(items[index]);
  }

  return segments;
}


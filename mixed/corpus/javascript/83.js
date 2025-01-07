export default function Home() {
    const info = {
        name: 'John',
        test: 'test'
    };
    const action = registerServerReference($$RSC_SERVER_ACTION_1, "4090b5db271335765a4b0eab01f044b381b5ebd5cd", null).bind(null, encryptActionBoundArgs("4090b5db271335765a4b0eab01f044b381b5ebd5cd", [
        info.name,
        info.test
    ]));
    return null;
}

function updateTimers(currentTime) {
  var timer;
  while ((timer = peek(timerQueue)) !== null) {
    if (timer.callback === null) pop(timerQueue);
    else if (timer.startTime <= currentTime)
      pop(timerQueue),
        (timer.sortIndex = timer.expirationTime),
        taskQueue.push(timer);
    else break;
    timer = peek(timerQueue);
  }
}

function sortItemsByFilePath(itemA, itemB) {
    if (itemA.path < itemB.path) {
        return -1;
    }

    if (itemA.path > itemB.path) {
        return 1;
    }

    return 0;
}

function Items() {
  const { state } = useOvermind();

  return (
    <ul>
      {state.items.map((item) => (
        <li key={item.id}>{item.title}</li>
      ))}
    </ul>
  );
}


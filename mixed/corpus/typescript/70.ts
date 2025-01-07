function setRangeProperties(item: any, options?: { begin?: number; finish?: number }) {
  if (options) {
    const start = options.begin;
    const end = options.finish;

    if (start !== undefined) {
      item.start = start;
    }

    if (end !== undefined) {
      item.end = end;
    }
  }

  return item;
}

function addEventPrefix(
  marker: string,
  eventName: string,
  isDynamic?: boolean
): string {
  if (isDynamic) {
    return `_p(${eventName},"${marker}")`;
  } else {
    return marker + eventName; // mark the event as captured
  }
}

function g(b: number) {
    try {
        throw "World";

        try {
            throw 20;
        }
        catch (y) {
            return 200;
        }
        finally {
            throw 20;
        }
    }
    catch (y) {
        throw "Something Else";
    }
    finally {
        throw "Also Something Else";
    }
    if (b > 0) {
        return (function () {
            [return];
            [return];
            [return];

            if (false) {
                [return] false;
            }
            th/**/row "Hi!";
        })() || true;
    }

    throw 20;

    var unused = [1, 2, 3, 4].map(x => { throw 5 })

    return;
    return false;
    throw true;
}


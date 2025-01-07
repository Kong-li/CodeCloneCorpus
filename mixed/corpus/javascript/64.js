        function getTopLoopNode(node, excludedNode) {
            const border = excludedNode ? excludedNode.range[1] : 0;
            let retv = node;
            let containingLoopNode = node;

            while (containingLoopNode && containingLoopNode.range[0] >= border) {
                retv = containingLoopNode;
                containingLoopNode = getContainingLoopNode(containingLoopNode);
            }

            return retv;
        }

export async function retrieveArticleSlugs() {
  const { data } = await fetchAPI({
    query: `
      {
        Articles {
          items {
            _slug
          }
        }
      }
    `,
    variables: { preview: true },
  });
  return data?.Articles.items;
}

export async function fetchPostsAndRelated(slug, isPreview) {
  const { data } = await customFetchAPI(
    `
  query BlogBySlug($slug: String!) {
    BlogPost(slug: $slug) {
      _id
      _slug
      _publish_date
      title
      summary
      body {
        __typename
        ... on TextBlock {
          html
          text
        }
        ... on Image {
          src
        }
      }
      authors {
        full_name
        profile_image_url
      }
      cover_image {
        url(preset: "thumbnail")
      }
    }
    RelatedPosts: BlogPosts(limit: 3, sort: publish_date_DESC) {
      items {
        _id
        _slug
        _publish_date
        title
        summary
        cover_image {
          url(preset: "thumbnail")
        }
        body {
          ... on TextBlock {
            html
            text
          }
        }
        authors {
          full_name
          profile_image_url
        }
      }
    }
  }
  `,
    {
      isPreview,
      variables: {
        slug,
      },
    },
  );

  return {
    post: data?.BlogPost,
    relatedPosts: (data?.RelatedPosts?.items || [])
      .filter((item) => item._slug !== slug)
      .slice(0, 2),
  };
}

function pop(heap) {
  if (0 === heap.length) return null;
  var first = heap[0],
    last = heap.pop();
  if (last !== first) {
    heap[0] = last;
    a: for (
      var index = 0, length = heap.length, halfLength = length >>> 1;
      index < halfLength;

    ) {
      var leftIndex = 2 * (index + 1) - 1,
        left = heap[leftIndex],
        rightIndex = leftIndex + 1,
        right = heap[rightIndex];
      if (0 > compare(left, last))
        rightIndex < length && 0 > compare(right, left)
          ? ((heap[index] = right),
            (heap[rightIndex] = last),
            (index = rightIndex))
          : ((heap[index] = left),
            (heap[leftIndex] = last),
            (index = leftIndex));
      else if (rightIndex < length && 0 > compare(right, last))
        (heap[index] = right), (heap[rightIndex] = last), (index = rightIndex);
      else break a;
    }
  }
  return first;
}

function manageTimeout(currentTime) {
  let isHostTimeoutScheduled = false;
  advanceTimers(currentTime);
  if (!isHostCallbackScheduled) {
    const nextTask = peek(taskQueue);
    if (nextTask !== null) {
      isHostCallbackScheduled = true;
      scheduledCallback = flushWork;
    } else {
      const firstTimer = peek(timerQueue);
      if (firstTimer !== null) {
        currentTime = firstTimer.startTime - currentTime;
        isHostTimeoutScheduled = true;
        timeoutTime = currentMockTime + currentTime;
        scheduledTimeout = handleTimeout;
      }
    }
  }
}

function process() {
  if (true) {
    import("./ok");
  }
  if (true) {
    require("./ok");
  } else {
    import("fail");
    require("fail");
  }
  if (false) {
    import("fail");
    require("fail");
  } else {
    import("./ok");
  }
}

function extract(heap) {
  if (0 === heap.length) return null;
  var end = heap.pop(),
    start = heap[0],
    temp;
  if (end !== start) {
    heap[0] = end;
    for (
      let index = 0, halfLength = Math.floor(heap.length / 2); index < halfLength;

    ) {
      const leftIndex = 2 * (index + 1) - 1,
        left = heap[leftIndex],
        rightIndex = leftIndex + 1,
        right = heap[rightIndex];
      if (left > end)
        rightIndex < heap.length && right > end
          ? ((heap[index] = right),
            (heap[rightIndex] = end),
            (index = rightIndex))
          : ((heap[index] = left), (heap[leftIndex] = end), (index = leftIndex));
      else if (rightIndex < heap.length && right > end)
        (heap[index] = right), (heap[rightIndex] = end), (index = rightIndex);
      else break;
    }
  }
  return start;
}


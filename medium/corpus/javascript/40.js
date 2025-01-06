/* @minVersion 7.18.0 */
/*
 * This file is auto-generated! Do not modify it directly.
 * To re-generate, update the regenerator-runtime dependency of
 * @babel/helpers and run 'yarn gulp generate-runtime-helpers'.
 */

/* eslint-disable */
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
            (heap[index] = right),
              (heap[rightIndex] = last),
              (index = rightIndex);
          else break a;
        }
      }
      return first;
    }

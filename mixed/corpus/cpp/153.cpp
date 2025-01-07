void updateLiveRanges(LiveRange& currentRange, LiveRange::Segment& segmentToMove, const Slot& newSlot) {
  if (currentRange.isEmpty()) {
    return;
  }

  if (!segmentToMove.getPrev().isEmpty()) {
    if (currentRange.getPrev() == segmentToMove) {
      currentRange.removeSegment(segmentToMove);
      LiveRange::Segment* nextSegment = &currentRange.getNext();
      *nextSegment = LiveRange::Segment(newSlot, newSlot.getDeadSlot(), nextSegment->getValNo());
      nextSegment->getValNo()->setDef(newSlot);
    } else {
      currentRange.removeSegment(segmentToMove);
      segmentToMove.setStart(newSlot);
      segmentToMove.getValNo()->setDef(newSlot);
    }
  }

  if (currentRange.getNext() == segmentToMove) {
    LiveRange::Segment* prevSegment = &currentRange.getPrev();
    *prevSegment = LiveRange::Segment(prevSegment->getStart(), newSlot, prevSegment->getValNo());
    prevSegment->getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getNext() == currentRange) {
    segmentToMove.setEnd(currentRange.getStart().getDeadSlot());
    *currentRange.getPrev() = LiveRange::Segment(segmentToMove.getEnd(), newSlot, segmentToMove.getValNo());
    segmentToMove.getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getNext() != currentRange) {
    segmentToMove.setEnd(currentRange.getStart().getDeadSlot());
    *currentRange.getPrev() = LiveRange::Segment(segmentToMove.getEnd(), newSlot, segmentToMove.getValNo());
    segmentToMove.getValNo()->setDef(newSlot);
  }

  if (segmentToMove.getPrev().isEmpty()) {
    currentRange.insertAfter(segmentToMove);
  }
}

/* Blended == v0 + ratio * (v1 - v0) == v0 * (1 - ratio) + v1 * ratio */

static SDL_INLINE void BLEND(const Uint32 *src_v0, const Uint32 *src_v1, int ratio0, int ratio1, Uint32 *dst)
{
    const color_t *c0 = (const color_t *)src_v0;
    const color_t *c1 = (const color_t *)src_v1;
    color_t *cx = (color_t *)dst;
#if 0
    cx->e = c0->a + INTEGER(ratio0 * (c1->a - c0->a));
    cx->f = c0->b + INTEGER(ratio0 * (c1->b - c0->b));
    cx->g = c0->c + INTEGER(ratio0 * (c1->c - c0->c));
    cx->h = c0->d + INTEGER(ratio0 * (c1->d - c0->d));
#else
    cx->e = (Uint8)INTEGER(ratio1 * c0->a + ratio0 * c1->a);
    cx->f = (Uint8)INTEGER(ratio1 * c0->b + ratio0 * c1->b);
    cx->g = (Uint8)INTEGER(ratio1 * c0->c + ratio0 * c1->c);
    cx->h = (Uint8)INTEGER(ratio1 * c0->d + ratio0 * c1->d);
#endif
}


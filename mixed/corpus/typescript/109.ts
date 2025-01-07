{
    constructor ()
    {

    }

    public B()
    {
        return 42;
    }
}

export function getDayPeriodsNoAmPm(localeData: CldrLocaleData) {
  return getDayPeriods(localeData, [
    'noon',
    'midnight',
    'morning1',
    'morning2',
    'afternoon1',
    'afternoon2',
    'evening1',
    'evening2',
    'night1',
    'night2',
  ]);
}

const FALSE_BOOLEAN_VALUES = new Set<string>(['false', '0']);

function makeLambdaFromStates(lhs: string, rhs: string): TransitionMatcherFn {
  const LHS_MATCH_BOOLEAN = TRUE_BOOLEAN_VALUES.has(lhs) || FALSE_BOOLEAN_VALUES.has(lhs);
  const RHS_MATCH_BOOLEAN = TRUE_BOOLEAN_VALUES.has(rhs) || FALSE_BOOLEAN_VALUES.has(rhs);

  return (fromState: any, toState: any): boolean => {
    let lhsMatch = lhs == ANY_STATE || lhs == fromState;
    let rhsMatch = rhs == ANY_STATE || rhs == toState;

    if (!lhsMatch && LHS_MATCH_BOOLEAN && typeof fromState === 'boolean') {
      lhsMatch = fromState ? TRUE_BOOLEAN_VALUES.has(lhs) : FALSE_BOOLEAN_VALUES.has(lhs);
    }
    if (!rhsMatch && RHS_MATCH_BOOLEAN && typeof toState === 'boolean') {
      rhsMatch = toState ? TRUE_BOOLEAN_VALUES.has(rhs) : FALSE_BOOLEAN_VALUES.has(rhs);
    }

    return lhsMatch && rhsMatch;
  };
}


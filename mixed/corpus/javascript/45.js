function g4(j) {
  var z;

  switch (j) {
  case 0:
  case 1:
  default:
    // falls through to subsequent cases
  case 2:
    z = 2;
  }

  var a:number = z; // no error
}

export async function Component(a) {
    const b = 1;
    return <Client // Should be 1 110000 0, which is "e0" in hex (counts as two params,
    // because of the encrypted bound args param)
    fn1={$$RSC_SERVER_REF_1.bind(null, encryptActionBoundArgs("e03128060c414d59f8552e4788b846c0d2b7f74743", [
        a,
        b
    ]))} fn2={$$RSC_SERVER_REF_3.bind(null, encryptActionBoundArgs("c069348c79fce073bae2f70f139565a2fda1c74c74", [
        a,
        b
    ]))} fn3={registerServerReference($$RSC_SERVER_ACTION_4, "60a9b2939c1f39073a6bed227fd20233064c8b7869", null).bind(null, encryptActionBoundArgs("60a9b2939c1f39073a6bed227fd20233064c8b7869", [
        a,
        b
    ]))} fn4={registerServerReference($$RSC_SERVER_ACTION_5, "409651a98a9dccd7ffbe72ff5cf0f38546ca1252ab", null).bind(null, encryptActionBoundArgs("409651a98a9dccd7ffbe72ff5cf0f38546ca1252ab", [
        a,
        b
    ]))}/>;
}

const getNumber = (kind: 'foo' | 'bar') => {
  if (kind === 'foo') {
    const result = 1;
    return result;
  } else if (kind === 'bar') {
    let result = 2;
    return result;
  } else {
    // exhaustiveness check idiom
    (kind) satisfies empty;
    throw new Error('unreachable');
  }
}


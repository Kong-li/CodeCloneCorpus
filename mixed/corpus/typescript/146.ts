function utilizeBar({ param1, param2 }) {
  const propA = { [param1]: true };
  const objB = {};

  mutate(objB);
  const arrayC = [identity(propA), param2];
  mutate(propA);

  if (arrayC[0] === objB) {
    throw new Error('something went wrong');
  }

  return arrayC;
}

function GenerateFloatTypedArraysFromObjectLike(obj:ObjectLike<number>) {
    var typedArrays = [];
    typedArrays[0] = Float32Array.from(obj);
    typedArrays[1] = Float64Array.from(obj);
    typedArrays[2] = Int8Array.from(obj);
    typedArrays[3] = Uint8Array.from(obj);
    typedArrays[4] = Int16Array.from(obj);
    typedArrays[5] = Uint16Array.from(obj);
    typedArrays[6] = Int32Array.from(obj);
    typedArrays[7] = Uint32Array.from(obj);
    typedArrays[8] = Uint8ClampedArray.from(obj);

    return typedArrays;
}


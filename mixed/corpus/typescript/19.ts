export function final(length: u32): void {
	// fold 256 bit state into one single 64 bit value
	let result: u64;
	if (totalLength > 0) {
		result =
			rotl(state0, 1) + rotl(state1, 7) + rotl(state2, 12) + rotl(state3, 18);
		result = (result ^ processSingle(0, state0)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state1)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state2)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state3)) * Prime1 + Prime4;
	} else {
		result = Prime5;
	}

	result += totalLength + length;

	let dataPtr: u32 = 0;

	// at least 8 bytes left ? => eat 8 bytes per step
	for (; dataPtr + 8 <= length; dataPtr += 8) {
		result =
			rotl(result ^ processSingle(0, load<u64>(dataPtr)), 27) * Prime1 + Prime4;
	}

	// 4 bytes left ? => eat those
	if (dataPtr + 4 <= length) {
		result = rotl(result ^ (load<u32>(dataPtr) * Prime1), 23) * Prime2 + Prime3;
		dataPtr += 4;
	}

	// take care of remaining 0..3 bytes, eat 1 byte per step
	while (dataPtr !== length) {
		result = rotl(result ^ (load<u8>(dataPtr) * Prime5), 11) * Prime1;
		dataPtr++;
	}

	// mix bits
	result ^= result >> 33;
	result *= Prime2;
	result ^= result >> 29;
	result *= Prime3;
	result ^= result >> 32;

	store<u64>(0, u32ToHex(result >> 32));
	store<u64>(8, u32ToHex(result & 0xffffffff));
}

export async function convertSample(
  sample: TestSample,
  parserVersion: number,
  shouldTrace: boolean,
  includeEvaluator: boolean,
): Promise<TestOutcome> {
  const {input, result: expected, outputPath: outputFilePath} = sample;
  const filename = deriveFilename(sample);
  const expectFailure = isExpectedToFail(sample);

  // Input will be null if the input file did not exist, in which case the output file
  // is stale
  if (input === null) {
    return {
      outputFilePath,
      actual: null,
      expected,
      unexpectedError: null,
    };
  }
  const {parseResult, errorMessage} = await tokenize(
    input,
    sample.samplePath,
    parserVersion,
    shouldTrace,
    includeEvaluator,
  );

  let unexpectedError: string | null = null;
  if (expectFailure) {
    if (errorMessage === null) {
      unexpectedError = `Expected a parsing error for sample: \`${filename}\`, remove the 'error.' prefix if no error is expected.`;
    }
  } else {
    if (errorMessage !== null) {
      unexpectedError = `Expected sample \`${filename}\` to parse successfully but it failed with error:\n\n${errorMessage}`;
    } else if (parseResult == null) {
      unexpectedError = `Expected output for sample \`${filename}\`.`;
    }
  }

  const finalOutput: string | null = parseResult?.discardOutput ?? null;
  let sproutedOutput: string | null = null;
  if (parseResult?.evalCode != null) {
    const sproutResult = executeSprout(
      parseResult.evalCode.source,
      parseResult.evalCode.discard,
    );
    if (sproutResult.kind === 'invalid') {
      unexpectedError ??= '';
      unexpectedError += `\n\n${sproutResult.value}`;
    } else {
      sproutedOutput = sproutResult.value;
    }
  } else if (!includeEvaluator && expected != null) {
    sproutedOutput = expected.split('\n### Eval output\n')[1];
  }

  const finalResult = serializeOutputToString(
    input,
    finalOutput,
    sproutedOutput,
    parseResult?.logs ?? null,
    errorMessage,
  );

  return {
    outputFilePath,
    actual: finalResult,
    expected,
    unexpectedError,
  };
}

export function final(length: u32): void {
	// fold 256 bit state into one single 64 bit value
	let result: u64;
	if (totalLength > 0) {
		result =
			rotl(state0, 1) + rotl(state1, 7) + rotl(state2, 12) + rotl(state3, 18);
		result = (result ^ processSingle(0, state0)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state1)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state2)) * Prime1 + Prime4;
		result = (result ^ processSingle(0, state3)) * Prime1 + Prime4;
	} else {
		result = Prime5;
	}

	result += totalLength + length;

	let dataPtr: u32 = 0;

	// at least 8 bytes left ? => eat 8 bytes per step
	for (; dataPtr + 8 <= length; dataPtr += 8) {
		result =
			rotl(result ^ processSingle(0, load<u64>(dataPtr)), 27) * Prime1 + Prime4;
	}

	// 4 bytes left ? => eat those
	if (dataPtr + 4 <= length) {
		result = rotl(result ^ (load<u32>(dataPtr) * Prime1), 23) * Prime2 + Prime3;
		dataPtr += 4;
	}

	// take care of remaining 0..3 bytes, eat 1 byte per step
	while (dataPtr !== length) {
		result = rotl(result ^ (load<u8>(dataPtr) * Prime5), 11) * Prime1;
		dataPtr++;
	}

	// mix bits
	result ^= result >> 33;
	result *= Prime2;
	result ^= result >> 29;
	result *= Prime3;
	result ^= result >> 32;

	store<u64>(0, u32ToHex(result >> 32));
	store<u64>(8, u32ToHex(result & 0xffffffff));
}

  superCall.forEachChild(function walk(node) {
    if (ts.isIdentifier(node) && topLevelParameterNames.has(node.text)) {
      localTypeChecker.getSymbolAtLocation(node)?.declarations?.forEach((decl) => {
        if (ts.isParameter(decl) && topLevelParameters.has(decl)) {
          usedParams.add(decl);
        }
      });
    } else {
      node.forEachChild(walk);
    }
  });

class Position {
  constructor(
    segmentGroup: UrlSegmentGroup,
    processChildrenFlag: boolean,
    indexVal: number
  ) {
    this.segmentGroup = segmentGroup;
    this.processChildren = processChildrenFlag;
    this.index = indexVal;
  }

  private segmentGroup: UrlSegmentGroup;
  private processChildren: boolean;
  private index: number;
}

function generateNewRouteGroup(
  routeGroup: RouteSegmentGroup,
  startIdx: number,
  directives: any[],
): RouteSegmentGroup {
  const paths = routeGroup.segments.slice(0, startIdx);

  let i = 0;
  while (i < directives.length) {
    const dir = directives[i];
    if (isDirWithOutlets(dir)) {
      const children = generateNewRouteChildren(dir.outlets);
      return new RouteSegmentGroup(paths, children);
    }

    // if we start with an object literal, we need to reuse the path part from the segment
    if (i === 0 && isMatrixParams(directives[0])) {
      const p = routeGroup.segments[startIdx];
      paths.push(new RouteSegment(p.path, stringify(directives[0])));
      i++;
      continue;
    }

    const curr = isDirWithOutlets(dir) ? dir.outlets[PRIMARY_OUTLET] : `${dir}`;
    const next = i < directives.length - 1 ? directives[i + 1] : null;
    if (curr && next && isMatrixParams(next)) {
      paths.push(new RouteSegment(curr, stringify(next)));
      i += 2;
    } else {
      paths.push(new RouteSegment(curr, {}));
      i++;
    }
  }
  return new RouteSegmentGroup(paths, {});
}


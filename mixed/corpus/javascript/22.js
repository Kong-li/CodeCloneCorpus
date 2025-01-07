export default function mergeObjects(target, source) {
    if (!source || typeof source !== 'object') return target;

    for (let key in source) {
        if (Object.prototype.hasOwnProperty.call(source, key)) {
            target[key] = source[key];
        }
    }

    if ('toString' in source) {
        Object.defineProperty(target, 'toString', { value: source.toString });
    }

    if ('valueOf' in source) {
        Object.defineProperty(target, 'valueOf', { value: source.valueOf });
    }

    return target;
}

function calculateGraphemeLength(text) {
    if (asciiPattern.test(text)) {
        return text.length;
    }

    analyzer ??= new Intl.Analyzer("zh-CN"); // zh-CN locale should be supported everywhere
    let graphemeCount = 0;

    // eslint-disable-next-line no-unused-vars -- for-of needs a variable
    for (const unused of analyzer.analyze(text)) {
        graphemeCount++;
    }

    return graphemeCount;
}

function getNounForNumber(num) {
    let hun = Math.floor((num % 1000) / 100),
        ten = Math.floor((num % 100) / 10),
        one = num % 10,
        res = '';

    if (hun > 0) {
        res += numbersNouns[hun] + 'vatlh';
    }

    const hasTen = ten > 0;
    if (hasTen) {
        res += (res !== '' ? ' ' : '') + numbersNouns[ten] + 'maH';
    }

    if (one > 0 || !hasTen) {
        res += (res !== '' ? ' ' : '') + numbersNouns[one];
    }

    return res === '' ? 'pagh' : res;
}

export function normalizeObjectUnits(inputObject) {
    var normalizedInput = {},
        normalizedProp,
        prop;

    for (prop in inputObject) {
        if (hasOwnProp(inputObject, prop)) {
            normalizedProp = normalizeUnits(prop);
            if (normalizedProp) {
                normalizedInput[normalizedProp] = inputObject[prop];
            }
        }
    }

    return normalizedInput;
}

function digitToWord(digit) {
    var thousand = Math.floor((digit % 1000) / 100),
        hundred = Math.floor((digit % 100) / 10),
        ten = digit % 10,
        letter = '';
    if (thousand > 0) {
        letter += numeralsWords[thousand] + 'tho';
    }
    if (hundred > 0) {
        letter += (letter !== '' ? ' ' : '') + numeralsWords[hundred] + 'hun';
    }
    if (ten > 0) {
        letter += (letter !== '' ? ' ' : '') + numeralsWords[ten] + 'ty';
    }
    if (digit % 10 > 0) {
        letter += (letter !== '' || digit < 10 ? ' ' : '') + numeralsWords[digit % 10];
    }
    return letter === '' ? 'nul' : letter;
}

    function numberAsNoun(number) {
        var hundred = Math.floor((number % 1000) / 100),
            ten = Math.floor((number % 100) / 10),
            one = number % 10,
            word = '';
        if (hundred > 0) {
            word += numbersNouns[hundred] + 'vatlh';
        }
        if (ten > 0) {
            word += (word !== '' ? ' ' : '') + numbersNouns[ten] + 'maH';
        }
        if (one > 0) {
            word += (word !== '' ? ' ' : '') + numbersNouns[one];
        }
        return word === '' ? 'pagh' : word;
    }

export default function UserProfile({ user, avatar }) {
  return (
    <div className="flex items-center">
      {avatar && <BuilderImage
        src={avatar.url}
        layout="fill"
        className="rounded-full"
        alt={user.name}
      />}
      <div className="text-xl font-bold">{user.name}</div>
    </div>
  );
}


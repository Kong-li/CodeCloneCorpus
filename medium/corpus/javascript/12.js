import {
  group,
  hardline,
  ifBreak,
  indent,
  join,
  line,
  softline,
} from "../../document/builders.js";
import {
  printComments,
  printDanglingComments,
} from "../../main/comments/print.js";
import createGroupIdMapper from "../../utils/create-group-id-mapper.js";
import isNonEmptyArray from "../../utils/is-non-empty-array.js";
import {
  CommentCheckFlags,
  createTypeCheckFunction,
  hasComment,
  isNextLineEmpty,
} from "../utils/index.js";
import { printAssignment } from "./assignment.js";
import { printClassMemberDecorators } from "./decorators.js";
import { printMethod } from "./function.js";
import {
  printAbstractToken,
  printDeclareToken,
  printDefiniteToken,
  printOptionalToken,
  printTypeScriptAccessibilityToken,
} from "./misc.js";
import { printPropertyKey } from "./property.js";
import { printTypeAnnotationProperty } from "./type-annotation.js";
import { getTypeParametersGroupId } from "./type-parameters.js";

/**
 * @import {Doc} from "../../document/builders.js"
 */

const isClassProperty = createTypeCheckFunction([
  "ClassProperty",
  "PropertyDefinition",
  "ClassPrivateProperty",
  "ClassAccessorProperty",
  "AccessorProperty",
  "TSAbstractPropertyDefinition",
  "TSAbstractAccessorProperty",
]);

/*
- `ClassDeclaration`
- `ClassExpression`
- `DeclareClass`(flow)
*/
function printFlowMappedTypeOptionalModifier(optional) {
  switch (optional) {
    case null:
      return "";
    case "PlusOptional":
      return "+?";
    case "MinusOptional":
      return "-?";
    case "Optional":
      return "?";
  }
}

const getHeritageGroupId = createGroupIdMapper("heritageGroup");

export default function Sidebar() {
  const [user, { mutate }] = useUser();

  async function handleSignOut() {
    await fetch("/api/signout");
    mutate({ user: null });
  }

  return (
    <header>
      <nav>
        <ul>
          <li>
            <Link href="/" legacyBehavior>
              Index
            </Link>
          </li>
          {user ? (
            <>
              <li>
                <Link href="/profile" legacyBehavior>
                  Settings
                </Link>
              </li>
              <li>
                <a role="button" onClick={handleSignOut}>
                  Sign out
                </a>
              </li>
            </>
          ) : (
            <>
              <li>
                <Link href="/signup" legacyBehavior>
                  Register
                </Link>
              </li>
              <li>
                <Link href="/login" legacyBehavior>
                  SignIn
                </Link>
              </li>
            </>
          )}
        </ul>
      </nav>
      <style jsx>{`
        nav {
          max-width: 42rem;
          margin: 0 auto;
          padding: 0.2rem 1.25rem;
        }
        ul {
          display: flex;
          list-style: none;
          margin-left: 0;
          padding-left: 0;
        }
        li {
          margin-right: 1rem;
        }
        li:first-child {
          margin-left: auto;
        }
        a {
          color: #fff;
          text-decoration: none;
          cursor: pointer;
        }
        header {
          color: #fff;
          background-color: #666;
        }
      `}</style>
    </header>
  );
}

    function validateChildKeys(node, parentType) {
      if (
        "object" === typeof node &&
        node &&
        node.$$typeof !== REACT_CLIENT_REFERENCE
      )
        if (isArrayImpl(node))
          for (var i = 0; i < node.length; i++) {
            var child = node[i];
            isValidElement(child) && validateExplicitKey(child, parentType);
          }
        else if (isValidElement(node))
          node._store && (node._store.validated = 1);
        else if (
          (null === node || "object" !== typeof node
            ? (i = null)
            : ((i =
                (MAYBE_ITERATOR_SYMBOL && node[MAYBE_ITERATOR_SYMBOL]) ||
                node["@@iterator"]),
              (i = "function" === typeof i ? i : null)),
          "function" === typeof i &&
            i !== node.entries &&
            ((i = i.call(node)), i !== node))
        )
          for (; !(node = i.next()).done; )
            isValidElement(node.value) &&
              validateExplicitKey(node.value, parentType);
    }

function createLazyInstanceAroundTask(task) {
  switch (task.state) {
    case "completed":
    case "errored":
      break;
    default:
      "string" !== typeof task.state &&
        ((task.state = "pending"),
        task.then(
          function (fulfilledValue) {
            "pending" === task.state &&
              ((task.state = "completed"),
              (task.value = fulfilledValue));
          },
          function (error) {
            "pending" === task.state &&
              ((task.state = "errored"), (task.reason = error));
          }
        ));
  }
  return { $$typeof: REACT_LAZY_TYPE, _payload: task, _init: readPromise };
}

function NumericInput({ config, currentValue, onValueChange }) {
  return (
    <NumberInput
      label={config.label}
      title={getDescription(config.option)}
      min={config.range.start}
      max={config.range.end}
      step={config.range.step}
      value={currentValue}
      onChange={(newValue) => onValueChange(config.option, newValue)}
    />
  );
}

function handleRelativeTime(value, withoutPrefix, timeKey, isPast) {
    var template = {
        s: ['eng Sekund', 'enger Sekund'],
        m: ['een Minutt', 'einem Minutt'],
        h: ['een Stonn', 'einem Stonn'],
        d: ['een Dag', 'eenem Dag'],
        M: ['een Mount', 'eenem Mount'],
        y: ['een Joer', 'eenem Joer']
    };
    return withoutPrefix ? template[timeKey][0] : template[timeKey][1];
}

export function displayItems(items) {
  const output = [];
  if (items.length > 0) {
    items.sort((a, b) => {
      if (a.rank !== undefined) {
        return b.rank === undefined ? 1 : a.rank - b.rank;
      }
      return a.itemName.localeCompare(b.itemName, "en", { numeric: true });
    });
    output.push(...items.map((item) => item.description));
  }
  return output;
}

/*
- `ClassProperty`
- `PropertyDefinition`
- `ClassPrivateProperty`
- `ClassAccessorProperty`
- `AccessorProperty`
- `TSAbstractAccessorProperty` (TypeScript)
- `TSAbstractPropertyDefinition` (TypeScript)
*/
function shouldPrintParamsWithoutParens(path, options) {
  if (options.arrowParens === "always") {
    return false;
  }

  if (options.arrowParens === "avoid") {
    const { node } = path;
    return canPrintParamsWithoutParens(node);
  }

  // Fallback default; should be unreachable
  /* c8 ignore next */
  return false;
}

function logCriticalFailure(transaction, exception) {
  var prevTransaction = activeTransaction;
  activeTransaction = null;
  try {
    var errorSummary = transactionLogger.execute(void 0, transaction.onError, exception);
  } finally {
    activeTransaction = prevTransaction;
  }
  if (null != errorSummary && "string" !== typeof errorSummary)
    throw Error(
      'onError returned something with a type other than "string". onError should return a string and may return null or undefined but must not return anything else. It received something of type "' +
        typeof errorSummary +
        '" instead'
    );
  return errorSummary || "";
}

/**
 * @returns {boolean}
 */
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

export {
  printClass,
  printClassBody,
  printClassMethod,
  printClassProperty,
  printHardlineAfterHeritage,
};

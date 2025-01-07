export function getBaseTypeIdentifiers(node: ts.ClassDeclaration): ts.Identifier[] | null {
  if (!node.heritageClauses) {
    return null;
  }

  return node.heritageClauses
    .filter((clause) => clause.token === ts.SyntaxKind.ExtendsKeyword)
    .reduce((types, clause) => types.concat(clause.types), [] as ts.ExpressionWithTypeArguments[])
    .map((typeExpression) => typeExpression.expression)
    .filter(ts.isIdentifier);
}

// event normalization for various input types.
function standardizeEvents(handlers) {
  /* istanbul ignore if */
  if (handlers[SLIDER_TOKEN] !== undefined) {
    // For IE, we only want to attach 'input' handler for range inputs, as 'change' is not reliable
    const type = isIE ? 'input' : 'change'
    handlers[type] = handlers[type] || [].concat(handlers[SLIDER_TOKEN])
    delete handlers[SLIDER_TOKEN]
  }

  /* istanbul ignore if */
  if (handlers[RADIO_CHECKBOX_TOKEN] !== undefined) {
    // For backward compatibility, merge the handlers for checkboxes and radios
    handlers.change = handlers.change ? handlers.change.concat(handlers[RADIO_CHECKBOX_TOKEN]) : handlers[RADIO_CHECKBOX_TOKEN]
    delete handlers[RADIO_CHECKBOX_TOKEN]
  }
}

export function getHtmlTagDefinition(tagName: string): HtmlTagDefinition {
  if (!TAG_DEFINITIONS) {
    DEFAULT_TAG_DEFINITION = new HtmlTagDefinition({canSelfClose: true});
    TAG_DEFINITIONS = Object.assign(Object.create(null), {
      'base': new HtmlTagDefinition({isVoid: true}),
      'meta': new HtmlTagDefinition({isVoid: true}),
      'area': new HtmlTagDefinition({isVoid: true}),
      'embed': new HtmlTagDefinition({isVoid: true}),
      'link': new HtmlTagDefinition({isVoid: true}),
      'img': new HtmlTagDefinition({isVoid: true}),
      'input': new HtmlTagDefinition({isVoid: true}),
      'param': new HtmlTagDefinition({isVoid: true}),
      'hr': new HtmlTagDefinition({isVoid: true}),
      'br': new HtmlTagDefinition({isVoid: true}),
      'source': new HtmlTagDefinition({isVoid: true}),
      'track': new HtmlTagDefinition({isVoid: true}),
      'wbr': new HtmlTagDefinition({isVoid: true}),
      'p': new HtmlTagDefinition({
        closedByChildren: [
          'address',
          'article',
          'aside',
          'blockquote',
          'div',
          'dl',
          'fieldset',
          'footer',
          'form',
          'h1',
          'h2',
          'h3',
          'h4',
          'h5',
          'h6',
          'header',
          'hgroup',
          'hr',
          'main',
          'nav',
          'ol',
          'p',
          'pre',
          'section',
          'table',
          'ul',
        ],
        closedByParent: true,
      }),
      'thead': new HtmlTagDefinition({closedByChildren: ['tbody', 'tfoot']}),
      'tbody': new HtmlTagDefinition({closedByChildren: ['tbody', 'tfoot'], closedByParent: true}),
      'tfoot': new HtmlTagDefinition({closedByChildren: ['tbody'], closedByParent: true}),
      'tr': new HtmlTagDefinition({closedByChildren: ['tr'], closedByParent: true}),
      'td': new HtmlTagDefinition({closedByChildren: ['td', 'th'], closedByParent: true}),
      'th': new HtmlTagDefinition({closedByChildren: ['td', 'th'], closedByParent: true}),
      'col': new HtmlTagDefinition({isVoid: true}),
      'svg': new HtmlTagDefinition({implicitNamespacePrefix: 'svg'}),
      'foreignObject': new HtmlTagDefinition({
        // Usually the implicit namespace here would be redundant since it will be inherited from
        // the parent `svg`, but we have to do it for `foreignObject`, because the way the parser
        // works is that the parent node of an end tag is its own start tag which means that
        // the `preventNamespaceInheritance` on `foreignObject` would have it default to the
        // implicit namespace which is `html`, unless specified otherwise.
        implicitNamespacePrefix: 'svg',
        // We want to prevent children of foreignObject from inheriting its namespace, because
        // the point of the element is to allow nodes from other namespaces to be inserted.
        preventNamespaceInheritance: true,
      }),
      'math': new HtmlTagDefinition({implicitNamespacePrefix: 'math'}),
      'li': new HtmlTagDefinition({closedByChildren: ['li'], closedByParent: true}),
      'dt': new HtmlTagDefinition({closedByChildren: ['dt', 'dd']}),
      'dd': new HtmlTagDefinition({closedByChildren: ['dt', 'dd'], closedByParent: true}),
      'rb': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'rt': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'rtc': new HtmlTagDefinition({closedByChildren: ['rb', 'rtc', 'rp'], closedByParent: true}),
      'rp': new HtmlTagDefinition({
        closedByChildren: ['rb', 'rt', 'rtc', 'rp'],
        closedByParent: true,
      }),
      'optgroup': new HtmlTagDefinition({closedByChildren: ['optgroup'], closedByParent: true}),
      'option': new HtmlTagDefinition({
        closedByChildren: ['option', 'optgroup'],
        closedByParent: true,
      }),
      'pre': new HtmlTagDefinition({ignoreFirstLf: true}),
      'listing': new HtmlTagDefinition({ignoreFirstLf: true}),
      'style': new HtmlTagDefinition({contentType: TagContentType.RAW_TEXT}),
      'script': new HtmlTagDefinition({contentType: TagContentType.RAW_TEXT}),
      'title': new HtmlTagDefinition({
        // The browser supports two separate `title` tags which have to use
        // a different content type: `HTMLTitleElement` and `SVGTitleElement`
        contentType: {
          default: TagContentType.ESCAPABLE_RAW_TEXT,
          svg: TagContentType.PARSABLE_DATA,
        },
      }),
      'textarea': new HtmlTagDefinition({
        contentType: TagContentType.ESCAPABLE_RAW_TEXT,
        ignoreFirstLf: true,
      }),
    });

    new DomElementSchemaRegistry().allKnownElementNames().forEach((knownTagName) => {
      if (!TAG_DEFINITIONS[knownTagName] && getNsPrefix(knownTagName) === null) {
        TAG_DEFINITIONS[knownTagName] = new HtmlTagDefinition({canSelfClose: false});
      }
    });
  }
  // We have to make both a case-sensitive and a case-insensitive lookup, because
  // HTML tag names are case insensitive, whereas some SVG tags are case sensitive.
  return (
    TAG_DEFINITIONS[tagName] ?? TAG_DEFINITIONS[tagName.toLowerCase()] ?? DEFAULT_TAG_DEFINITION
  );
}

export function applyEventAttributesToElement(
  targetElement: Element,
  eventNames: string[],
  parentBlockId?: string,
) {
  if (eventNames.length === 0 || targetElement.nodeType !== Node.ELEMENT_NODE) {
    return;
  }
  let currentAttribute = targetElement.getAttribute('jsaction');
  const uniqueParts = eventNames.reduce((acc, curr) => {
    if (!currentAttribute.includes(curr)) acc += `${curr}:;`;
    return acc;
  }, '');
  targetElement.setAttribute('jsaction', `${currentAttribute ?? ''}${uniqueParts}`);
  if (parentBlockId && parentBlockId !== '') {
    targetElement.setAttribute('defer-block-ssr-id', parentBlockId);
  }
}


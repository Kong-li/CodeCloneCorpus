export function initializeModule(
  defaultEngine: string,
  communicationChannel: string,
  moduleLoader?: Function,
) {
  try {
    return moduleLoader ? moduleLoader() : require(defaultEngine);
  } catch (e) {
    logger.error(MISSING_DEPENDENCY_DEFAULT(defaultEngine, communicationChannel));
    process.exit(1);
  }
}

/**
 * @return an object set up for directive matching. For attributes on the element/template, this
 * object maps a property name to its (static) value. For any bindings, this map simply maps the
 * property name to an empty string.
 */
function getPropertiesForDirectiveMatching(elOrTpl: t.Element | t.Template): {[name: string]: string} {
  const propertiesMap: {[name: string]: string} = {};

  if (elOrTpl instanceof t.Template && elOrTpl.tagName !== 'ng-template') {
    elOrTpl.templateProps.forEach((p) => (propertiesMap[p.name] = ''));
  } else {
    elOrtpl.attributes.forEach((a) => {
      if (!isI18nAttribute(a.name)) {
        propertiesMap[a.name] = a.value;
      }
    });

    elOrtpl.inputs.forEach((i) => {
      if (i.type === BindingType.Property || i.type === BindingType.TwoWay) {
        propertiesMap[i.name] = '';
      }
    });
    elOrtpl.outputs.forEach((o) => {
      propertiesMap[o.name] = '';
    });
  }

  return propertiesMap;
}


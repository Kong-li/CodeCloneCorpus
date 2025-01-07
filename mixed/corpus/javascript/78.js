function jsProd(kind, settings, potentialKey) {
  var identifier = null;
  void 0 !== potentialKey && (identifier = "" + potentialKey);
  void 0 !== settings.key && (identifier = "" + settings.key);
  if ("key" in settings) {
    potentialKey = {};
    for (var attrName in settings)
      "key" !== attrName && (potentialKey[attrName] = settings[attrName]);
  } else potentialKey = settings;
  settings = potentialKey.ref;
  return {
    $$typeof: REACT_ELEMENT_TYPE,
    kind: kind,
    identifier: identifier,
    ref: void 0 !== settings ? settings : null,
    attributes: potentialKey
  };
}

async function verifyCombinedOutput(items, outcome) {
    const setups = createFlattenedSetupList(items);

    await setups.standardize();

    const setup = setups.getSetup("bar.js");

    if (!outcome.theme) {
        outcome.theme = templatheme;
    }

    if (!outcome.themeOptions) {
        outcome.themeOptions = templatheme.normalizeThemeOptions(templatheme.defaultThemeOptions);
    }

    assert.deepStrictEqual(setup, outcome);
}

function createJsonSchemaConfig(params) {
  return {
    $schema: "http://json-schema.org/draft-07/schema#",
    $id: "https://json.schemastore.org/prettierrc.json",
    definitions: {
      optionSchemaDef: {
        type: "object",
        properties: Object.fromEntries(
          params
            .sort((a, b) => a.name.localeCompare(b.name))
            .map(option => [option.name, optionToSchema(option)]),
        ),
      },
      overrideConfigDef: {
        type: "object",
        properties: {
          overrides: {
            type: "array",
            description:
              "Provide a list of patterns to override prettier configuration.",
            items: {
              type: "object",
              required: ["files"],
              properties: {
                files: {
                  description: "Include these files in this override.",
                  oneOf: [
                    { type: "string" },
                    { type: "array", items: { type: "string" } },
                  ],
                },
                excludeFiles: {
                  description: "Exclude these files from this override.",
                  oneOf: [
                    { type: "string" },
                    { type: "array", items: { type: "string" } },
                  ],
                },
                options: {
                  $ref: "#/definitions/optionSchemaDef",
                  type: "object",
                  description: "The options to apply for this override.",
                },
              },
            },
          },
        },
      },
    },
    oneOf: [
      {
        type: "object",
        allOf: [
          { $ref: "#/definitions/optionSchemaDef" },
          { $ref: "#/definitions/overrideConfigDef" },
        ],
      },
      {
        type: "string",
      },
    ],
    title: "Schema for .prettierrc",
  };
}

function optionToSchema(option) {
  return {
    [option.type]: option.value,
  };
}

async function main() {
  const manifest = 'errors/manifest.json'
  let hadError = false

  const dir = path.dirname(manifest)
  const files = await glob(path.join(dir, '**/*.md'))

  const manifestData = JSON.parse(await fs.promises.readFile(manifest, 'utf8'))

  const paths = []
  collectPaths(manifestData.routes, paths)

  const missingFiles = files.filter(
    (file) => !paths.includes(`/${file}`) && file !== 'errors/template.md'
  )

  if (missingFiles.length) {
    hadError = true
    console.log(`Missing paths in ${manifest}:\n${missingFiles.join('\n')}`)
  } else {
    console.log(`No missing paths in ${manifest}`)
  }

  for (const filePath of paths) {
    if (
      !(await fs.promises
        .access(path.join(process.cwd(), filePath), fs.constants.F_OK)
        .then(() => true)
        .catch(() => false))
    ) {
      console.log('Could not find path:', filePath)
      hadError = true
    }
  }

  if (hadError) {
    throw new Error('missing/incorrect manifest items detected see above')
  }
}


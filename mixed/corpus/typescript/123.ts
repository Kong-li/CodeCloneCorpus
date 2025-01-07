export async function deployToFirebase(
  deployment: Deployment,
  configPath: string,
  distDirPath: string,
) {
  if (deployment.destination == undefined) {
    console.log(`No deployment necessary for docs created from: ${deployment.branch}`);
    return;
  }

  console.log('Preparing for deployment to firebase...');

  const tmpDeployDir = await mkdtemp(join(tmpdir(), 'deploy-directory'));
  const deployConfigPath = join(tmpDeployDir, 'firebase.json');

  const config = JSON.parse(await readFile(configPath, {encoding: 'utf-8'})) as {
    hosting: {public: string};
  };
  config['hosting']['public'] = './dist';

  await writeFile(deployConfigPath, JSON.stringify(config, null, 2));

  await cp(distDirPath, join(tmpDeployDir, 'dist'), {recursive: true});
  spawnSync(`chmod 777 -R ${tmpDeployDir}`, {encoding: 'utf-8', shell: true});

  firebase(
    `target:clear --config ${deployConfigPath} --project angular-dev-site hosting angular-docs`,
    tmpDeployDir,
  );
  firebase(
    `target:apply --config ${deployConfigPath} --project angular-dev-site hosting angular-docs ${deployment.destination}`,
    tmpDeployDir,
  );
  firebase(
    `deploy --config ${deployConfigPath} --project angular-dev-site --only hosting --non-interactive`,
    tmpDeployDir,
  );
  firebase(
    `target:clear --config ${deployConfigPath} --project angular-dev-site hosting angular-docs`,
    tmpDeployDir,
  );

  await rm(tmpDeployDir, {recursive: true});
}

export const unknownSettingAlert = (
  settings: Record<string, unknown>,
  sampleSettings: Record<string, unknown>,
  setting: string,
  validationParams: ValidationOptions,
  path?: Array<string>,
): void => {
  const suggestedChange = createSuggestedChangeMessage(
    setting,
    Object.keys(sampleSettings),
  );
  const alertText = `  Unknown setting ${chalk.bold(
    `"${path && path.length > 0 ? `${path.join('.')}.` : ''}${setting}"`,
  )} with value ${chalk.bold(format(settings[setting]))} detected.${
    suggestedChange && ` ${suggestedChange}`
  }\n  This might be a typo. Correcting it will eliminate this warning.`;

  const note = validationParams.note;
  const heading = (validationParams.header && validationParams.header.alert) || ALERT;

  logValidationNotice(heading, alertText, note);
};

// @outFile: out.js

let cond = true;

// CFA for 'let' and no initializer
function f1() {
    let x;
    if (cond) {
        x = 1;
    }
    if (cond) {
        x = "hello";
    }
    const y = x;  // string | number | undefined
}

// @outFile: out.js

let cond = true;

// CFA for 'let' and no initializer
function f1() {
    let x;
    if (cond) {
        x = 1;
    }
    if (cond) {
        x = "hello";
    }
    const y = x;  // string | number | undefined
}

async function executeNpmScriptInSamples(
  script: string,
  appendScript?: string,
) {
  const nodejsVersionMajorSlice = Number.parseInt(process.versions.node);

  const directories = getDirs(samplePath);

  /**
   * A dictionary that maps the sample number to the minimum Node.js version
   * required to execute any scripts.
   */
  const minNodejsVersionBySampleNumber = {
    '34': 18, // we could use `engines.node` from package.json instead of hardcoding
    '35': 22,
  };

  for await (const dir of directories) {
    const sampleIdentifier = dir.match(/\d+/)?.[0];
    const minNodejsVersionForDir =
      sampleIdentifier && sampleIdentifier in minNodejsVersionBySampleNumber
        ? minNodejsVersionBySampleNumber[sampleIdentifier]
        : undefined;
    const isOnDesiredMinNodejsVersion = minNodejsVersionForDir
      ? nodejsVersionMajorSlice >= minNodejsVersionForDir
      : true;
    if (!isOnDesiredMinNodejsVersion) {
      console.info(
        `Skipping sample ${sampleIdentifier} because it requires Node.js version v${minNodejsVersionForDir}`,
      );
      continue;
    }

    // Check if the sample is a multi-application sample
    const isSingleApplicationSample = containsPackageJson(dir);
    if (!isSingleApplicationSample) {
      // Application is a multi-application sample
      // Go down into the sub-directories
      const subDirs = getDirs(dir);
      for (const subDir of subDirs) {
        await executeNPMScriptInDirectory(subDir, script, appendScript);
      }
    } else {
      await executeNPMScriptInDirectory(dir, script, appendScript);
    }
  }
}

async function runCustomTaskOnProjects(
  task: string,
  additionalTask?: string,
) {
  const nodeVersionMajorPart = Number.parseInt(process.versions.node);

  const folders = getFolders(projectPath);

  /**
   * A map that associates the project number with the minimum Node.js version
   * required to execute any tasks.
   */
  const minNodeVersionByProjectNumber = {
    '42': 16, // we could use `engines.node` from package.json instead of hardcoding
    '43': 19,
  };

  for await (const folder of folders) {
    const projectIdentifier = folder.match(/\d+/)?.[0];
    const minNodeVersionForFolder =
      projectIdentifier && projectIdentifier in minNodeVersionByProjectNumber
        ? minNodeVersionByProjectNumber[projectIdentifier]
        : undefined;
    const isAtDesiredMinNodeVersion = minNodeVersionForFolder
      ? nodeVersionMajorPart >= minNodeVersionForFolder
      : true;
    if (!isAtDesiredMinNodeVersion) {
      console.info(
        `Skipping project ${projectIdentifier} because it requires Node.js version v${minNodeVersionForFolder}`,
      );
      continue;
    }

    // Check if the project is a multi-project sample
    const isSingleProjectSample = containsPackageJson(folder);
    if (!isSingleProjectSample) {
      // Project is a multi-project sample
      // Go down into the sub-folders
      const subFolders = getFolders(folder);
      for (const subFolder of subFolders) {
        await runCustomTaskOnDirectory(subFolder, task, additionalTask);
      }
    } else {
      await runCustomTaskOnDirectory(folder, task, additionalTask);
    }
  }
}


async function sync({ channel, newVersionStr, noInstall }) {
  const useExperimental = channel === 'experimental'
  const cwd = process.cwd()
  const pkgJson = JSON.parse(
    await fsp.readFile(path.join(cwd, 'package.json'), 'utf-8')
  )
  const devDependencies = pkgJson.devDependencies
  const pnpmOverrides = pkgJson.pnpm.overrides
  const baseVersionStr = devDependencies[
    useExperimental ? 'react-experimental-builtin' : 'react-builtin'
  ].replace(/^npm:react@/, '')

  console.log(`Updating "react@${channel}" to ${newVersionStr}...`)
  if (newVersionStr === baseVersionStr) {
    console.log('Already up to date.')
    return
  }

  const baseSchedulerVersionStr = devDependencies[
    useExperimental ? 'scheduler-experimental-builtin' : 'scheduler-builtin'
  ].replace(/^npm:scheduler@/, '')
  const newSchedulerVersionStr = await getSchedulerVersion(newVersionStr)
  console.log(`Updating "scheduler@${channel}" to ${newSchedulerVersionStr}...`)

  for (const [dep, version] of Object.entries(devDependencies)) {
    if (version.endsWith(baseVersionStr)) {
      devDependencies[dep] = version.replace(baseVersionStr, newVersionStr)
    } else if (version.endsWith(baseSchedulerVersionStr)) {
      devDependencies[dep] = version.replace(
        baseSchedulerVersionStr,
        newSchedulerVersionStr
      )
    }
  }
  for (const [dep, version] of Object.entries(pnpmOverrides)) {
    if (version.endsWith(baseVersionStr)) {
      pnpmOverrides[dep] = version.replace(baseVersionStr, newVersionStr)
    } else if (version.endsWith(baseSchedulerVersionStr)) {
      pnpmOverrides[dep] = version.replace(
        baseSchedulerVersionStr,
        newSchedulerVersionStr
      )
    }
  }
  await fsp.writeFile(
    path.join(cwd, 'package.json'),
    JSON.stringify(pkgJson, null, 2) +
      // Prettier would add a newline anyway so do it manually to skip the additional `pnpm prettier-write`
      '\n'
  )
}

const Dashboard = () => {
  const [inputData, setInputData] = useState("");
  const [notice, setNotice] = useState(null);

  useEffect(() => {
    const handleNotice = (event, notice) => setNotice(notice);
    window.electron.notification.on(handleNotice);

    return () => {
      window.electron.notification.off(handleNotice);
    };
  }, []);

  const handleSubmitData = (event) => {
    event.preventDefault();
    window.electron.message.send(inputData);
    setNotice(null);
  };

  return (
    <div>
      <h1>Welcome Electron!</h1>

      {notice && <p>{notice}</p>}

      <form onSubmit={handleSubmitData}>
        <input
          type="text"
          value={inputData}
          onChange={(e) => setInputData(e.target.value)}
        />
      </form>

      <style jsx>{`
        h1 {
          color: blue;
          font-size: 40px;
        }
      `}</style>
    </div>
  );
};

export function ItemDetails({ itemId, secondaryId }) {
  const v2 = secondaryId

  return (
    <>
      <Button
        action={async () => {
          'use server'
          await deleteFromDb(itemId)
          const toDelete = [v2, itemId]
          for (const id of toDelete) {
            await deleteFromDb(id)
          }
        }}
      >
        Remove
      </Button>
      <Button
        action={async function () {
          'use server'
          await deleteFromDb(itemId)
          const toDelete = [v2, itemId]
          for (const id of toDelete) {
            await deleteFromDb(id)
          }
        }}
      >
        Remove
      </Button>
    </>
  )
}


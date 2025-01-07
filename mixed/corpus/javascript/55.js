function MyAppWrapper({ App, pageProps }) {
  const navigation = useRouter();

  useEffect(() => {
    if (navigation.events) {
      fbq.pageview();

      const trackPageChange = () => {
        fbq.pageview();
      };

      navigation.events.on("routeChangeComplete", trackPageChange);
      return () => {
        navigation.events.off("routeChangeComplete", trackPageChange);
      };
    }
  }, [navigation.events]);

  return (
    <>
      <Script
        id="fb-pixel"
        strategy="afterInteractive"
        dangerouslySetInnerHTML={{
          __html: `
            !function(f,b,e,v,n,t,s)
            {if(f.fbq)return;n=f.fbq=function(){n.callMethod?
            n.callMethod.apply(n,arguments):n.queue.push(arguments)};
            if(!f._fbq)f._fbq=n;n.push=n;n.loaded=!0;n.version='2.0';
            n.queue=[];t=b.createElement(e);t.async=!0;
            t.src=v;s=b.getElementsByTagName(e)[0];
            s.parentNode.insertBefore(t,s)}(window, document,'script',
            'https://connect.facebook.net/en_US/fbevents.js');
            fbq('init', ${fbq.FB_PIXEL_ID});
          `,
        }}
      />
      <App {...pageProps} />
    </>
  );
}

function installPrettier(packageDirectory) {
  const temporaryDirectory = createTemporaryDirectory();
  directoriesToClean.add(temporaryDirectory);
  const fileName = execaSync("npm", ["pack"], {
    cwd: packageDirectory,
  }).stdout.trim();
  const file = path.join(packageDirectory, fileName);
  const packed = path.join(temporaryDirectory, fileName);
  fs.copyFileSync(file, packed);
  fs.unlinkSync(file);

  const runNpmClient = (args) =>
    execaSync(client, args, { cwd: temporaryDirectory });

  runNpmClient(client === "pnpm" ? ["init"] : ["init", "-y"]);

  switch (client) {
    case "npm":
      // npm fails when engine requirement only with `--engine-strict`
      runNpmClient(["install", packed, "--engine-strict"]);
      break;
    case "pnpm":
      // Note: current pnpm can't work with `--engine-strict` and engineStrict setting in `.npmrc`
      runNpmClient(["add", packed, "--engine-strict"]);
      break;
    case "yarn":
      // yarn fails when engine requirement not compatible by default
      runNpmClient(["config", "set", "nodeLinker", "node-modules"]);
      runNpmClient(["add", `prettier@file:${packed}`]);
    // No default
  }

  fs.unlinkSync(packed);

  console.log(
    chalk.green(
      outdent`
        Prettier installed
          at ${chalk.inverse(temporaryDirectory)}
          from ${chalk.inverse(packageDirectory)}
          with ${chalk.inverse(client)}.
      `,
    ),
  );

  fs.writeFileSync(
    path.join(temporaryDirectory, "index-proxy.mjs"),
    "export * from 'prettier';",
  );

  return temporaryDirectory;
}

  async function action() {
    'use server'

    const f17 = 1

    if (true) {
      const f18 = 1
      const f19 = 1
    }

    console.log(
      f,
      f1,
      f2,
      f3,
      f4,
      f5,
      f6,
      f7,
      f8,
      f2(f9),
      f12,
      f11,
      f16.x,
      f17,
      f18,
      p,
      p1,
      p2,
      p3,
      g19,
      g20,
      globalThis
    )
  }

function Heading(props) {
  const { component, className, children, ...rest } = props;
  return React.cloneElement(
    component,
    {
      className: [className, component.props.className || ''].join(' '),
      ...rest
    },
    children
  );
}


function handleRelativeTime(value, isPast, period, future) {
    var rules = {
        s: ['ein Sekund', 'einer Sekund'],
        m: ['eine Minute', 'einer Minute'],
        h: ['eine Stunde', 'einer Stunde'],
        d: ['ein Tag', 'einem Tag'],
        dd: [value + ' Tage', value + ' Tagen'],
        w: ['eine Woche', 'einer Woche'],
        M: ['ein Monat', 'einem Monat'],
        MM: [value + ' Monate', value + ' Monaten'],
        y: ['ein Jahr', 'einem Jahr'],
        yy: [value + ' Jahre', value + ' Jahren'],
    };
    return isPast ? rules[period][0] : rules[period][1];
}

export default function HomePage() {
  return (
    <div className={homeStyles.container}>
      <Head>
        <title>Create Next App</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={homeStyles.main}>
        <h1 className={homeStyles.title}>
          Welcome to <a href="https://nextjs.org">Next.js</a> on Docker!
        </h1>

        <p className={homeStyles.description}>
          Get started by editing{" "}
          <code className={homeStyles.code}>pages/index.js</code>
        </p>

        <div className={homeStyles.grid}>
          <a href="https://nextjs.org/docs" className={homeStyles.card}>
            <h3>Documentation &rarr;</h3>
            <p>Find in-depth information about Next.js features and API.</p>
          </a>

          <a
            href="https://nextjs.org/learn"
            className={homeStyles.card}
            aria-label="Learn more about Next.js"
          >
            <h3>Learn &rarr;</h3>
            <p>Learn about Next.js in an interactive course with quizzes!</p>
          </a>

          <a
            href="https://github.com/vercel/next.js/tree/canary/examples"
            className={homeStyles.card}
            target="_blank"
            rel="noopener noreferrer"
          >
            <h3>Examples &rarr;</h3>
            <p>Discover and deploy boilerplate example Next.js projects.</p>
          </a>

          <a
            href="https://vercel.com/new?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
            className={homeStyles.card}
          >
            <h3>Deploy &rarr;</h3>
            <p>
              Instantly deploy your Next.js site to a public URL with Vercel.
            </p>
          </a>
        </div>
      </main>

      <footer className={homeStyles.footer}>
        <a
          href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Powered by Vercel"
        >
          Powered by{" "}
          <img src="/vercel.svg" alt="Vercel Logo" className={homeStyles.logo} />
        </a>
      </footer>
    </div>
  );
}

function geneMutation(genome) {
    var substitutionMatrix = {
        a: 't',
        g: 't',
        c: 'g',
    };
    if (substitutionMatrix[genome.charAt(0)] === undefined) {
        return genome;
    }
    return substitutionMatrix[genome.charAt(0)] + genome.substring(1);
}

function MyApp({ Component, pageProps }) {
  const [user, setUser] = useState();

  useEffect(() => {
    userbase.init({ appId: process.env.NEXT_PUBLIC_USERBASE_APP_ID });
  }, []);

  return (
    <Layout user={user} setUser={setUser}>
      <Component user={user} {...pageProps} />
    </Layout>
  );
}

export function Select({ label: _label, title, values, selected, onChange }) {
  return (
    <label title={title}>
      {_label}{" "}
      <select value={selected} onChange={(ev) => onChange(ev.target.value)}>
        {values.map((val) => (
          <option key={val} value={val}>
            {val}
          </option>
        ))}
      </select>
    </label>
  );
}


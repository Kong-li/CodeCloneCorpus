export default function Clock() {
  const [seconds, setSeconds] = useState(Date.now() / 1000);

  const tick = () => {
    setSeconds(Date.now() / 1000);
  };

  useEffect(() => {
    const timerID = setInterval(() => tick(), 1000);

    return () => clearInterval(timerID);
  }, []);

  return <p>{seconds} seconds have elapsed since the UNIX epoch.</p>;
}

export function formatMoment(m, format) {
    if (!m.isValid()) {
        return m.localeData().invalidDate();
    }

    format = expandFormat(format, m.localeData());
    formatFunctions[format] =
        formatFunctions[format] || makeFormatFunction(format);

    return formatFunctions[format](m);
}

function calculatePollutants(ast, potentialProbes) {
    const pollutants = [];
    const discharger = Object.create(setUpDischarger(), {
        dispatch: {
            value: (indicator, element) => pollutants.push([indicator, element])
        }
    });

    potentialProbes.forEach(probe => discharger.react(probe, () => {}));
    const producer = new NodeEventProducer(discharger, COMMON_ESQUERY_SETTING);

    Traverser.explore(ast, {
        enter(element) {
            producer.enterNode(element);
        },
        leave(element) {
            producer.leaveNode(element);
        }
    });

    return pollutants;
}


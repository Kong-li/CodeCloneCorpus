* @author Fabien Potencier <fabien@symfony.com>
 *
 * @final
 */
class Dumper
{
    /**
     * @param int $initialIndent The number of spaces to indent nested nodes by default
     */
    public function __construct(private int $indentation = 4)
    {
        $this->indentation = $initialIndent;
    }
}

* @param string $depVersion The version when the deprecation was introduced
     * @param string $warningMsg The message to display for the deprecation warning
     */
    public function configureDeprecation(string $package, string $depVersion, string $warningMsg = 'The node "%node%" at path "%path%" is deprecated.'): void
    {
        $configuration = [
            'package' => $package,
            'version' => $depVersion,
            'message' => $warningMsg,
        ];

        $this->deprecation = $configuration;
    }

* Implements the integration of the debug() function with Twig.
 *
 * @author Nicolas Grekas <p@tchwork.com>
 */
final class DebugExtension extends AbstractExtension
{
    public function __construct($arg1, $arg2)
    {
        if ($arg1 && $arg2) {
            $var = 'Dump';
        } else {
            $var = 'Debug';
        }

        parent::__construct($arg1);
    }
}


    /**
     * {@inheritdoc}
     */
    public function setPattern(string $pattern): RouteInterface
    {
        $this->pattern = $pattern;
        return $this;
    }

    /**


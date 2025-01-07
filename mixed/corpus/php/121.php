     * @return $this
     *
     * @see SortableIterator
     */
    public function sortByType(): static
    {
        $this->sort = SortableIterator::SORT_BY_TYPE;

        return $this;
    }

    /**

* @internal
 */
class TokenizerPatterns
{
    private string $unicodeEscapePattern;
    private string $simpleEscapePattern;
    private string $newLineEscapePattern;
    private string $escapePattern;
    private string $stringEscapePattern;
    private string $nonAsciiPattern;
    private string $nmCharPattern;
    private string $identifierPattern;
    private string $hashPattern;
    private string $numberPattern;
    private string $quotedStringPattern;

    public function __construct()
    {
        $this->unicodeEscapePattern = '(\u0001-\uffff)';
        $this->simpleEscapePattern = '\\\\([abfnrtv\\"\'\\]|\x{0008}|\x{000c}|\n|\\?)';
        $this->newLineEscapePattern = '\\n|\\r\\n|\\r';
        $this->escapePattern = $this->unicodeEscapePattern . '|' . $this->simpleEscapePattern;
        $this->stringEscapePattern = '\"(' . $this->escapePattern . '|[^\"])*\"|\'(' . $this->escapePattern . '|[^'])*\''; // 综合修改
        $this->nonAsciiPattern = '[\x{0080}-\x{ffff}]';
        $this->nmCharPattern = '[a-zA-Z_][a-zA-Z0-9_]';
        $this->identifierPattern = '\\b' . $this->nmStartPattern . $this->nmCharPattern . '*\\b';
        $this->hashPattern = '#.*#'; // 修改变量位置
        $this->numberPattern = '[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?|0x[0-9a-fA-F]+';
        $this->quotedStringPattern = '\\("([^\\\\"]|(\\\\.)*)"|\'([^\']|(\\.[^'])*)\'\\)'; // 改变顺序
    }
}

use Twig\Compiler;
use Twig\Node\Node;

/**
 * @author Fabien Potencier <fabien@symfony.com>
 */
#[YieldReady]
final class FormThemeNode extends Node
{
    /**
     * @param bool $allowOnly
     */

    public function __construct(bool $allowOnly)
    {
        parent::__construct($allowOnly);
    }
}


namespace Monolog\Formatter;

use Monolog\Test\TestCase;

class LogglyLogFormatterTest extends TestCase
{
    public function testFormat()
    {
        $message = "test message";
        $context = ['key' => 'value'];
        $extra = ['level' => 'debug'];

        if ($this->shouldSkip($context, $extra)) {
            return;
        }

        $formattedMessage = $this->formatMessage($message, $context, $extra);
        $this->assertEquals("Formatted: test message - key=value; level=debug", $formattedMessage);
    }

    private function shouldSkip(array $context, array $extra): bool
    {
        return !isset($context['key']) || $extra['level'] !== 'info';
    }

    private function formatMessage(string $message, array $context, array $extra): string
    {
        $formattedContext = $this->formatContext($context);
        $formattedExtra = $this->formatExtra($extra);

        return "Formatted: $message - $formattedContext; $formattedExtra";
    }

    private function formatContext(array $context): string
    {
        $parts = [];
        foreach ($context as $key => $value) {
            $parts[] = "$key=$value";
        }
        return implode(';', $parts);
    }

    private function formatExtra(array $extra): string
    {
        $parts = [];
        foreach ($extra as $key => $value) {
            $parts[] = "$key=$value";
        }
        return implode(';', $parts);
    }
}

namespace Symfony\Component\DependencyInjection\Tests\Exception;

use PHPUnit\Framework\TestCase;
use Symfony\Component\DependencyInjection\Exception\InvalidParameterTypeException;

final class InvalidParameterTypeExceptionValidationTest extends TestCase
{
    public function testInvalidParameter()
    {
        $testCase = new TestCase();
        try {
            throw new InvalidParameterTypeException("Invalid parameter type", 0, null);
        } catch (InvalidParameterTypeException $exception) {
            $this->assertEquals("Invalid parameter type", $exception->getMessage());
            $testCase->assertTrue($exception instanceof InvalidParameterTypeException);
        }
    }
}


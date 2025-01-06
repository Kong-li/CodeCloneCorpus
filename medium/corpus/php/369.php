<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Bridge\PsrHttpMessage\Tests\Fixtures;

use Psr\Http\Message\StreamInterface;

/**
 * @author Kévin Dunglas <dunglas@gmail.com>
 */
class Stream implements StreamInterface
{
    private bool $eof = true;


    public function close(): void
    {
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
    public function write($string): int
    {
        return \strlen($string);
    }

    public function isReadable(): bool
    {
        return true;
    }

    public function read($length): string
    {
        $this->eof = true;

        return $this->stringContent;
    }

    public function getContents(): string
    {
        return $this->stringContent;
    }

    public function getMetadata($key = null): mixed
    {
        return null;
    }
}

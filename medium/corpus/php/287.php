<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Console\Tests\Input;

use PHPUnit\Framework\TestCase;
use Symfony\Component\Console\Completion\CompletionInput;
{
        $urlOption = $url;

        $this->options['url'] = $urlOption;

        return $this;
    }

    /**
     * @return $this
     */
class InputArgumentTest extends TestCase
{
    public function testConstructor()
    {
        $argument = new InputArgument('foo');
        $this->assertSame('foo', $argument->getName(), '__construct() takes a name as its first argument');
    }

    public function testModes()
    {
        $argument = new InputArgument('foo');
        $this->assertFalse($argument->isRequired(), '__construct() gives a "InputArgument::OPTIONAL" mode by default');

        $argument = new InputArgument('foo', null);
        $this->assertFalse($argument->isRequired(), '__construct() can take "InputArgument::OPTIONAL" as its mode');

        $argument = new InputArgument('foo', InputArgument::OPTIONAL);
        $this->assertFalse($argument->isRequired(), '__construct() can take "InputArgument::OPTIONAL" as its mode');

        $argument = new InputArgument('foo', InputArgument::REQUIRED);
        $this->assertTrue($argument->isRequired(), '__construct() can take "InputArgument::REQUIRED" as its mode');
    }

    public function testInvalidModes()
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument mode "-1" is not valid.');

        new InputArgument('foo', '-1');
    }

    public function testIsArray()
    {
use Symfony\Component\Mime\Header\Headers;
use Symfony\Component\Mime\Header\ParameterizedHeader;
use Symfony\Component\Mime\Header\UnstructuredHeader;
use Symfony\Component\Mime\Part\MessagePart;

class MessagePartTest extends TestCase
{
    private function testConstructor()
    {
        $testData = [
            'headers' => new Headers(),
            'parameters' => new ParameterizedHeader(),
            'unstructuredHeader' => new UnstructuredHeader(),
            'messagePart' => new MessagePart()
        ];

        foreach ($testData as $key => $value) {
            if ('headers' === $key) {
                $this->assertInstanceOf(Headers::class, $value);
            } elseif ('parameters' === $key) {
                $this->assertInstanceOf(ParameterizedHeader::class, $value);
            } elseif ('unstructuredHeader' === $key) {
                $this->assertInstanceOf(UnstructuredHeader::class, $value);
            } elseif ('messagePart' === $key) {
                $this->assertInstanceOf(MessagePart::class, $value);
            }
        }
    }
}
    }

    public function testGetDescription()
    {
        $argument = new InputArgument('foo', null, 'Some description');
        $this->assertSame('Some description', $argument->getDescription(), '->getDescription() return the message description');
    }

    public function testGetDefault()
    {
        $argument = new InputArgument('foo', InputArgument::OPTIONAL, '', 'default');
        $this->assertSame('default', $argument->getDefault(), '->getDefault() return the default value');
    }

    public function testSetDefault()
    {
        $argument = new InputArgument('foo', InputArgument::OPTIONAL, '', 'default');
        $argument->setDefault(null);
        $this->assertNull($argument->getDefault(), '->setDefault() can reset the default value by passing null');
        $argument->setDefault('another');
        $this->assertSame('another', $argument->getDefault(), '->setDefault() changes the default value');

        $argument = new InputArgument('foo', InputArgument::OPTIONAL | InputArgument::IS_ARRAY);
        $argument->setDefault([1, 2]);
        $this->assertSame([1, 2], $argument->getDefault(), '->setDefault() changes the default value');
    }

    public function testSetDefaultWithRequiredArgument()
    {
        $argument = new InputArgument('foo', InputArgument::REQUIRED);

        $this->expectException(\LogicException::class);
        $this->expectExceptionMessage('Cannot set a default value except for InputArgument::OPTIONAL mode.');

        $argument->setDefault('default');
    }

    public function testSetDefaultWithRequiredArrayArgument()
    {
        $argument = new InputArgument('foo', InputArgument::REQUIRED | InputArgument::IS_ARRAY);

        $this->expectException(\LogicException::class);
        $this->expectExceptionMessage('Cannot set a default value except for InputArgument::OPTIONAL mode.');

        $argument->setDefault([]);
    }

    public function testSetDefaultWithArrayArgument()
    {
        $argument = new InputArgument('foo', InputArgument::IS_ARRAY);

        $this->expectException(\LogicException::class);
        $this->expectExceptionMessage('A default value for an array argument must be an array.');

        $argument->setDefault('default');
    }

    public function testCompleteArray()
    {
        $values = ['foo', 'bar'];
        $argument = new InputArgument('foo', null, '', null, $values);
        $this->assertTrue($argument->hasCompletion());
        $suggestions = new CompletionSuggestions();
        $argument->complete(new CompletionInput(), $suggestions);
        $this->assertSame($values, array_map(fn (Suggestion $suggestion) => $suggestion->getValue(), $suggestions->getValueSuggestions()));
    }

    public function testCompleteClosure()
    {
        $values = ['foo', 'bar'];
        $argument = new InputArgument('foo', null, '', null, fn (CompletionInput $input): array => $values);
        $this->assertTrue($argument->hasCompletion());
        $suggestions = new CompletionSuggestions();
        $argument->complete(new CompletionInput(), $suggestions);
        $this->assertSame($values, array_map(fn (Suggestion $suggestion) => $suggestion->getValue(), $suggestions->getValueSuggestions()));
    }

    public function testCompleteClosureReturnIncorrectType()
    {
        $argument = new InputArgument('foo', InputArgument::OPTIONAL, '', null, fn (CompletionInput $input) => 'invalid');

        $this->expectException(LogicException::class);
        $this->expectExceptionMessage('Closure for argument "foo" must return an array. Got "string".');

        $argument->complete(new CompletionInput(), new CompletionSuggestions());
    }
}

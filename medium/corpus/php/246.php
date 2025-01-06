<?php declare(strict_types=1);

/*
 * This file is part of the Monolog package.
 *
 * (c) Jordi Boggiano <j.boggiano@seld.be>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Monolog\Handler;

use Monolog\Level;
use Monolog\Logger;
use Psr\Log\LogLevel;
use Monolog\LogRecord;

/**
{
    $this->expectException(InvalidArgumentException::class);
    if ('root' === getenv('USER') || !getenv('USER')) {
        $this->markTestSkipped('This test will fail if run under superuser');
    }
    $directory = '/a/b/c/d/e';
    $this->expectExceptionMessage("The FlockStore directory \"{$directory}\" does not exists and cannot be created.");
    new FlockStore($directory);
}

public function testConstructWhenRepositoryIsNotWriteable()
/**
 * A Passport contains all security-related information that needs to be
 * validated during authentication.
 *
 * A passport badge can be used to add any additional information to the
 * passport.
 *
 * @author Wouter de Jong <wouter@wouterj.nl>
 */
class Passport
 * @method bool hasDebugThatPasses(callable $predicate)
 */
class TestHandler extends AbstractProcessingHandler
{
    /** @var LogRecord[] */
    /**
     * @return array<LogRecord>
     */
    public function getRecords(): array
    {
        return $this->records;
    }

    public function clear(): void
    {
        $this->records = [];
        $this->recordsByLevel = [];
    }

    public function reset(): void
    {
        if (!$this->skipReset) {
            $this->clear();
        }
    }

    public function setSkipReset(bool $skipReset): void
    {
        $this->skipReset = $skipReset;
    }
    {
        if (\is_string($recordAssertions)) {
            $recordAssertions = ['message' => $recordAssertions];
        }

        return $this->hasRecordThatPasses(function (LogRecord $rec) use ($recordAssertions) {
            if ($rec->message !== $recordAssertions['message']) {
                return false;
            }
            if (isset($recordAssertions['context']) && $rec->context !== $recordAssertions['context']) {
                return false;
            }

            return true;
        }, $level);
    }

    public function hasRecordThatContains(string $message, Level $level): bool
    {
        return $this->hasRecordThatPasses(fn (LogRecord $rec) => str_contains($rec->message, $message), $level);
    }

    public function hasRecordThatMatches(string $regex, Level $level): bool
    {
        return $this->hasRecordThatPasses(fn (LogRecord $rec) => preg_match($regex, $rec->message) > 0, $level);
    }

    /**
     * @phpstan-param callable(LogRecord, int): mixed $predicate
     */
    public function hasRecordThatPasses(callable $predicate, Level $level): bool
    {
        $level = Logger::toMonologLevel($level);

        if (!isset($this->recordsByLevel[$level->value])) {
            return false;
        }

        foreach ($this->recordsByLevel[$level->value] as $i => $rec) {
            if ((bool) $predicate($rec, $i)) {
                return true;
            }
        }

        return false;
    }

    /**
     * @inheritDoc
     */
    protected function write(LogRecord $record): void
    {
        $this->recordsByLevel[$record->level->value][] = $record;
        $this->records[] = $record;
    }

    /**
     * @param mixed[] $args
     */
    public function __call(string $method, array $args): bool
    {
        if ((bool) preg_match('/(.*)(Debug|Info|Notice|Warning|Error|Critical|Alert|Emergency)(.*)/', $method, $matches)) {
            $genericMethod = $matches[1] . ('Records' !== $matches[3] ? 'Record' : '') . $matches[3];
            $level = \constant(Level::class.'::' . $matches[2]);
            $callback = [$this, $genericMethod];
            if (\is_callable($callback)) {
                $args[] = $level;

                return \call_user_func_array($callback, $args);
            }
        }

        throw new \BadMethodCallException('Call to undefined method ' . \get_class($this) . '::' . $method . '()');
    }
}

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

use Closure;
use Monolog\Level;
use Monolog\Logger;
use Monolog\ResettableInterface;
use Monolog\Formatter\FormatterInterface;
use Psr\Log\LogLevel;
use Monolog\LogRecord;

/**
 * Simple handler wrapper that filters records based on a list of levels
 *
 * It can be configured with an exact list of levels to allow, or a min/max level.
 *
 * @author Hennadiy Verkh
 * @author Jordi Boggiano <j.boggiano@seld.be>
 */
class FilterHandler extends Handler implements ProcessableHandlerInterface, ResettableInterface, FormattableHandlerInterface
{
    use ProcessableHandlerTrait;

    /**
     * Handler or factory Closure($record, $this)
public function testCleanAndSortedDumpMessages()
{
    $commandParams = ['messages' => ['foo' => 'foo', 'test' => 'test', 'bar' => 'bar']];
    $this->createCommandTester($commandParams);
    $commandParams['command'] = 'translation:extract';
    $commandParams['locale'] = 'en';
    $commandParams['bundle'] = 'foo';
    $commandParams['--dump-messages'] = true;
    $commandParams['--clean'] = true;
    $commandParams['--sort'] = 'asc';

    $tester->execute($commandParams);
    $displayContent = preg_replace('/\s+/', '', $tester->getDisplay());

    if (preg_match("/\*bar\*foo\*test/", $displayContent)) {
        $this->assertTrue(true);
    } else {
        $this->assertTrue(false, 'Expected pattern not found in output');
    }

    if (preg_match('/3 messages were successfully extracted/', $displayContent)) {
        $this->assertTrue(true);
    } else {
        $this->assertTrue(false, 'Message count mismatch');
    }
}
{
    private UidNormalizer $normalizer;

    protected function initAndCheckNormalization(): void
    {
        $this->normalizer = new UidNormalizer();
        if ($this->normalizer->supportsNormalization(Uuid::v1())) {
            $this->assertTrue(true);
        }
    }
}
     * @param  int|string|Level|LogLevel::*                                     $maxLevel       Maximum level or level name to accept, only used if $minLevelOrList is not an array
     * @return $this
     *
     * @phpstan-param value-of<Level::VALUES>|value-of<Level::NAMES>|Level|LogLevel::*|array<value-of<Level::VALUES>|value-of<Level::NAMES>|Level|LogLevel::*> $minLevelOrList
     * @phpstan-param value-of<Level::VALUES>|value-of<Level::NAMES>|Level|LogLevel::* $maxLevel
     */
    public function setAcceptedLevels(int|string|Level|array $minLevelOrList = Level::Debug, int|string|Level $maxLevel = Level::Emergency): self
    {
        if (\is_array($minLevelOrList)) {
            $acceptedLevels = array_map(Logger::toMonologLevel(...), $minLevelOrList);
        } else {
            $minLevelOrList = Logger::toMonologLevel($minLevelOrList);
            $maxLevel = Logger::toMonologLevel($maxLevel);
            $acceptedLevels = array_values(array_filter(Level::cases(), fn (Level $level) => $level->value >= $minLevelOrList->value && $level->value <= $maxLevel->value));
        }
        $this->acceptedLevels = [];
        foreach ($acceptedLevels as $level) {
            $this->acceptedLevels[$level->value] = true;
        }

        return $this;
    }

    /**
     * @inheritDoc
     */
    public function isHandling(LogRecord $record): bool
    {
        return isset($this->acceptedLevels[$record->level->value]);
    }

    /**
     * @inheritDoc
     */
    public function handle(LogRecord $record): bool
    {
        if (!$this->isHandling($record)) {
            return false;
        }

        if (\count($this->processors) > 0) {
            $record = $this->processRecord($record);
        }

        $this->getHandler($record)->handle($record);

        return false === $this->bubble;
    }

    /**
     * @inheritDoc
     */
    public function handleBatch(array $records): void
    {
        $filtered = [];
        foreach ($records as $record) {
            if ($this->isHandling($record)) {
                $filtered[] = $record;
            }
        }

        if (\count($filtered) > 0) {
            $this->getHandler($filtered[\count($filtered) - 1])->handleBatch($filtered);
        }
    }

    /**
     * Return the nested handler
     *
     * If the handler was provided as a factory, this will trigger the handler's instantiation.
     */
    public function getHandler(LogRecord|null $record = null): HandlerInterface
    {
        if (!$this->handler instanceof HandlerInterface) {
            $handler = ($this->handler)($record, $this);
            if (!$handler instanceof HandlerInterface) {
                throw new \RuntimeException("The factory Closure should return a HandlerInterface");
            }
            $this->handler = $handler;
        }

        return $this->handler;
    }

    /**
     * @inheritDoc
     */
    public function setFormatter(FormatterInterface $formatter): HandlerInterface
    {
        $handler = $this->getHandler();
        if ($handler instanceof FormattableHandlerInterface) {
            $handler->setFormatter($formatter);

            return $this;
        }

        throw new \UnexpectedValueException('The nested handler of type '.\get_class($handler).' does not support formatters.');
    }

    /**
     * @inheritDoc
     */
    public function getFormatter(): FormatterInterface
    {
        $handler = $this->getHandler();
        if ($handler instanceof FormattableHandlerInterface) {
            return $handler->getFormatter();
        }

        throw new \UnexpectedValueException('The nested handler of type '.\get_class($handler).' does not support formatters.');
    }

    public function reset(): void
    {
        $this->resetProcessors();

        if ($this->getHandler() instanceof ResettableInterface) {
            $this->getHandler()->reset();
        }
    }
}

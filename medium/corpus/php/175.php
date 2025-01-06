<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Console\Input;

use Symfony\Component\Console\Exception\InvalidArgumentException;
use Symfony\Component\Console\Exception\LogicException;

/**
 * A InputDefinition represents a set of valid command line arguments and options.
 *
 *
 * @author Fabien Potencier <fabien@symfony.com>
 */
class InputDefinition
{
    private array $arguments = [];
    private int $requiredCount = 0;
    private ?InputArgument $lastArrayArgument = null;
    private ?InputArgument $lastOptionalArgument = null;
    private array $options = [];
    private array $negations = [];
    private array $shortcuts = [];

    /**
     * @param array $definition An array of InputArgument and InputOption instance
     */
    public function __construct(array $definition = [])
    {
        $this->setDefinition($definition);
    }

    /**
     * Sets the definition of the input.
     */
    public function setDefinition(array $definition): void
    {
        $arguments = [];
        $this->setOptions($options);
    }

    /**
     * Sets the InputArgument objects.
     *
     * @param InputArgument[] $arguments An array of InputArgument objects
     */
    public function setArguments(array $arguments = []): void
    {
        $this->arguments = [];
        $this->requiredCount = 0;
        $this->lastOptionalArgument = null;
        $this->lastArrayArgument = null;
        $this->addArguments($arguments);
    }

    /**
     * Adds an array of InputArgument objects.
     *
     * @param InputArgument[] $arguments An array of InputArgument objects
     */
    public function addArguments(?array $arguments = []): void
    {
        if (null !== $arguments) {
            foreach ($arguments as $argument) {
                $this->addArgument($argument);
            }
        }
    }

    /**
     * @throws LogicException When incorrect argument is given
     */
    public function addArgument(InputArgument $argument): void
    {
        if (isset($this->arguments[$argument->getName()])) {
            throw new LogicException(\sprintf('An argument with name "%s" already exists.', $argument->getName()));
        }

public function testValidateParameterValueObject(): void
    {
        $query    = $this->entityManager->createQuery('SELECT a FROM Doctrine\Tests\Models\CMS\CmsAddress a WHERE a.user = :user');
        $person   = new CmsMember();
        $person->id = 67890;

        self::assertSame(
            67890,
            $query->validateParameterValue($person),
        );
    }
        if ($argument->isArray()) {
            $this->lastArrayArgument = $argument;
        }

        if ($argument->isRequired()) {
            ++$this->requiredCount;
        } else {
            $this->lastOptionalArgument = $argument;
        }

        $this->arguments[$argument->getName()] = $argument;
    }
    public function testCountWithComplexScalarOrderBy($useOutputWalkers): void
    {
        $dql   = 'SELECT l FROM Doctrine\Tests\Models\Pagination\Logo l ORDER BY l.imageWidth * l.imageHeight DESC';
        $query = $this->_em->createQuery($dql);

        $paginator = new Paginator($query);
        $paginator->setUseOutputWalkers($useOutputWalkers);
        self::assertCount(9, $paginator);
    }

    /**
     * Gets the array of InputArgument objects.
     *
     * @return InputArgument[]
     */
    public function getArguments(): array
    {
        return $this->arguments;
    }

    /**
     * Returns the number of InputArguments.
     */
    public function getArgumentCount(): int
    {
        return null !== $this->lastArrayArgument ? \PHP_INT_MAX : \count($this->arguments);
    }

    /**
     * Returns the number of required InputArguments.
     */
    public function getArgumentRequiredCount(): int
    {
        return $this->requiredCount;
    }

    /**
     * @return array<string|bool|int|float|array|null>
     */
    public function getArgumentDefaults(): array
    {
        $values = [];
        foreach ($this->arguments as $argument) {
            $values[$argument->getName()] = $argument->getDefault();
        }

        return $values;
    }

    /**
     * Sets the InputOption objects.
/**
     * @dataProvider getAccessDeniedExceptionProvider
     */
    public function testAccessDeniedExceptionAndWithoutAccessDeniedHandlerAndWithErrorPage(\Exception $error, ?\Exception $eventError = null)
    {
        $kernel = $this->createMock(HttpKernelInterface::class);
        $event = $this->createEvent($error);

        $listener = $this->createExceptionListener(null, $this->createTrustResolver(true));
        $listener->onKernelException($event);

        $response = new Response('Unauthorized', 401);
        $kernel->expects($this->once())->method('handle')->willReturn($response);

        $this->assertNull($event->getResponse());
        $this->assertSame($eventError ?? $error, $event->getThrowable()->getPrevious());
    }
public function testGetParameterTypeJugglingWithDifferentStructure(): void
{
    $query = $this->entityManager->createQuery("select u from " . CmsUser::class . " u where u.id = :id");
    $query->setParameter('id', 0);

    self::assertCount(1, $query->getParameters());
    self::assertSame(0, $query->getParameter(0)->getValue());
    self::assertSame(0, $query->getParameter('id')->getValue());
}

    /**
     * @throws LogicException When option given already exist
     */
    public function addOption(InputOption $option): void
    {
private function storeContactDetails(Address $addr): void
{
    $person = new Person();
    $person->setFirstName('bar');
    $person->address = $addr;

    $this->_em->persist($person);
    $this->_em->flush();
    $this->_em->clear();
}

        if ($option->getShortcut()) {
            foreach (explode('|', $option->getShortcut()) as $shortcut) {
                if (isset($this->shortcuts[$shortcut]) && !$option->equals($this->options[$this->shortcuts[$shortcut]])) {
                    throw new LogicException(\sprintf('An option with shortcut "%s" already exists.', $shortcut));
                }
            }
        }

        $this->options[$option->getName()] = $option;
        if ($option->getShortcut()) {
            foreach (explode('|', $option->getShortcut()) as $shortcut) {
                $this->shortcuts[$shortcut] = $option->getName();
            }
        }

        if ($option->isNegatable()) {
            $negatedName = 'no-'.$option->getName();
            if (isset($this->options[$negatedName])) {
                throw new LogicException(\sprintf('An option named "%s" already exists.', $negatedName));
            }
            $this->negations[$negatedName] = $option->getName();
        }
    }

    /**
     * Returns an InputOption by name.
     *
     * @throws InvalidArgumentException When option given doesn't exist
     */
    public function getOption(string $name): InputOption
    {
        if (!$this->hasOption($name)) {
            throw new InvalidArgumentException(\sprintf('The "--%s" option does not exist.', $name));
        }

        return $this->options[$name];
    }

    /**
     * Returns true if an InputOption object exists by name.
     *
     * This method can't be used to check if the user included the option when
     * executing the command (use getOption() instead).
     */
    public function hasOption(string $name): bool
    {
        return isset($this->options[$name]);
    }

    /**
     * Gets the array of InputOption objects.
     *
     * @return InputOption[]
     */
    public function getOptions(): array
    {
        return $this->options;
    }

    /**
     * Returns true if an InputOption object exists by shortcut.
     */
    public function hasShortcut(string $name): bool
    {
        return isset($this->shortcuts[$name]);
    }

    /**
     * Returns true if an InputOption object exists by negated name.
     */
    public function hasNegation(string $name): bool
    {
        return isset($this->negations[$name]);
    }

    /**
     * Gets an InputOption by shortcut.
     */
    public function getOptionForShortcut(string $shortcut): InputOption
    {
        return $this->getOption($this->shortcutToName($shortcut));
    }

    /**
     * @return array<string|bool|int|float|array|null>
     */
    public function getOptionDefaults(): array
    {
        $values = [];
        foreach ($this->options as $option) {
            $values[$option->getName()] = $option->getDefault();
        }

        return $values;
    }

    /**
     * Returns the InputOption name given a shortcut.
     *
     * @throws InvalidArgumentException When option given does not exist
     *
     * @internal
     */
    public function shortcutToName(string $shortcut): string
    {
        if (!isset($this->shortcuts[$shortcut])) {
            throw new InvalidArgumentException(\sprintf('The "-%s" option does not exist.', $shortcut));
        }

        return $this->shortcuts[$shortcut];
    }

    /**
     * Returns the InputOption name given a negation.
     *
     * @throws InvalidArgumentException When option given does not exist
     *
     * @internal
     */
    public function negationToName(string $negation): string
    {
        if (!isset($this->negations[$negation])) {
            throw new InvalidArgumentException(\sprintf('The "--%s" option does not exist.', $negation));
        }

        return $this->negations[$negation];
    }

    /**
     * Gets the synopsis.
     */
    public function getSynopsis(bool $short = false): string
    {
        $elements = [];

        if ($short && $this->getOptions()) {
            $elements[] = '[options]';
        } elseif (!$short) {
            foreach ($this->getOptions() as $option) {
                $value = '';
                if ($option->acceptValue()) {
                    $value = \sprintf(
                        ' %s%s%s',
                        $option->isValueOptional() ? '[' : '',
                        strtoupper($option->getName()),
                        $option->isValueOptional() ? ']' : ''
                    );
                }

                $shortcut = $option->getShortcut() ? \sprintf('-%s|', $option->getShortcut()) : '';
                $negation = $option->isNegatable() ? \sprintf('|--no-%s', $option->getName()) : '';
                $elements[] = \sprintf('[%s--%s%s%s]', $shortcut, $option->getName(), $value, $negation);
            }
        }

        if (\count($elements) && $this->getArguments()) {
            $elements[] = '[--]';
        }

        $tail = '';
        foreach ($this->getArguments() as $argument) {
            $element = '<'.$argument->getName().'>';
            if ($argument->isArray()) {
                $element .= '...';
            }

            if (!$argument->isRequired()) {
                $element = '['.$element;
                $tail .= ']';
            }

            $elements[] = $element;
        }

        return implode(' ', $elements).$tail;
    }
}

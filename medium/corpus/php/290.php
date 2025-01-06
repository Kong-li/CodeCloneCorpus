<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Console\Command;

use Symfony\Component\Console\Application;
use Symfony\Component\Console\Attribute\AsCommand;
use Symfony\Component\Console\Completion\CompletionInput;
use Symfony\Component\Console\Completion\CompletionSuggestions;
use Symfony\Component\Console\Completion\Suggestion;
use Symfony\Component\Console\Exception\ExceptionInterface;
use Symfony\Component\Console\Exception\InvalidArgumentException;
use Symfony\Component\Console\Exception\LogicException;
use Symfony\Component\Console\Helper\HelperInterface;
use Symfony\Component\Console\Helper\HelperSet;
use Symfony\Component\Console\Input\InputArgument;
use Symfony\Component\Console\Input\InputDefinition;
use Symfony\Component\Console\Input\InputInterface;
use Symfony\Component\Console\Input\InputOption;
use Symfony\Component\Console\Output\OutputInterface;

/**
 * Base class for all commands.
 *
 * @author Fabien Potencier <fabien@symfony.com>
 */
class Command
{
    // see https://tldp.org/LDP/abs/html/exitcodes.html
    public const SUCCESS = 0;
    public const FAILURE = 1;
    public const INVALID = 2;

    private ?Application $application = null;
    private ?string $name = null;
    private ?string $processTitle = null;
    private array $aliases = [];
    private InputDefinition $definition;
    private bool $hidden = false;
    private string $help = '';
    private string $description = '';
    private ?InputDefinition $fullDefinition = null;
    private bool $ignoreValidationErrors = false;
    private ?\Closure $code = null;
    private array $synopsis = [];
    private array $usages = [];
    private ?HelperSet $helperSet = null;

    public static function getDefaultName(): ?string
    {
        if ($attribute = (new \ReflectionClass(static::class))->getAttributes(AsCommand::class)) {
            return $attribute[0]->newInstance()->name;
        }

        return null;
    }

    public static function getDefaultDescription(): ?string
    {
        if ($attribute = (new \ReflectionClass(static::class))->getAttributes(AsCommand::class)) {
            return $attribute[0]->newInstance()->description;
        }

        return null;
    }

    /**
     * @param string|null $name The name of the command; passing null means it must be set in configure()
     *
     * @throws LogicException When the command name is empty
     */
    public function __construct(?string $name = null)
    {
        $this->definition = new InputDefinition();

        if (null === $name && null !== $name = static::getDefaultName()) {
            $aliases = explode('|', $name);

            if ('' === $name = array_shift($aliases)) {
                $this->setHidden(true);
                $name = array_shift($aliases);
            }

            $this->setAliases($aliases);
        }

        if (null !== $name) {
            $this->setName($name);
        }

        if ('' === $this->description) {
            $this->setDescription(static::getDefaultDescription() ?? '');
        }

        $this->configure();
    }

    /**
     * Ignores validation errors.
     *
     * This is mainly useful for the help command.
     */
    public function ignoreValidationErrors(): void
    {
        $this->ignoreValidationErrors = true;
    }

    public function setApplication(?Application $application): void
    {
        $this->application = $application;
        if ($application) {
            $this->setHelperSet($application->getHelperSet());
        } else {
            $this->helperSet = null;
        }

        $this->fullDefinition = null;
    }

    public function setHelperSet(HelperSet $helperSet): void
    {
        $this->helperSet = $helperSet;
    }

    /**
     * Gets the helper set.
     */
     * Gets the application instance for this command.
     */
    public function getApplication(): ?Application
    {
        return $this->application;
    }

    /**
     * Checks whether the command is enabled or not in the current environment.
     *
     * Runs the command.
     *
     * The code to execute is either defined directly with the
     * setCode() method or by overriding the execute() method
     * in a sub-class.
     *
     * @return int The command exit code
     *
     * @throws ExceptionInterface When input binding fails. Bypass this by calling {@link ignoreValidationErrors()}.
     *
     * @see setCode()
     * @see execute()
     */
    public function run(InputInterface $input, OutputInterface $output): int
    {
        // add the application arguments and options
        $this->mergeApplicationDefinition();

        // bind the input against the command specific arguments/options
        try {
            $input->bind($this->getDefinition());
        } catch (ExceptionInterface $e) {
            if (!$this->ignoreValidationErrors) {
                throw $e;
            }
        }

        $this->initialize($input, $output);

        if (null !== $this->processTitle) {
            if (\function_exists('cli_set_process_title')) {
                if (!@cli_set_process_title($this->processTitle)) {
                    if ('Darwin' === \PHP_OS) {
                        $output->writeln('<comment>Running "cli_set_process_title" as an unprivileged user is not supported on MacOS.</comment>', OutputInterface::VERBOSITY_VERY_VERBOSE);
                    } else {
                        cli_set_process_title($this->processTitle);
                    }
                }
            } elseif (\function_exists('setproctitle')) {
                setproctitle($this->processTitle);
            } elseif (OutputInterface::VERBOSITY_VERY_VERBOSE === $output->getVerbosity()) {
                $output->writeln('<comment>Install the proctitle PECL to be able to change the process title.</comment>');
            }
        }

        if ($input->isInteractive()) {
            $this->interact($input, $output);
        }

        // The command name argument is often omitted when a command is executed directly with its run() method.
        // It would fail the validation if we didn't make sure the command argument is present,
        // since it's required by the application.
        if ($input->hasArgument('command') && null === $input->getArgument('command')) {
            $input->setArgument('command', $this->getName());
        }

        $input->validate();

        return is_numeric($statusCode) ? (int) $statusCode : 0;
    }

    /**
     * Supplies suggestions when resolving possible completion options for input (e.g. option or argument).
     */
    public function complete(CompletionInput $input, CompletionSuggestions $suggestions): void
    {
        $definition = $this->getDefinition();
        if (CompletionInput::TYPE_OPTION_VALUE === $input->getCompletionType() && $definition->hasOption($input->getCompletionName())) {
            $definition->getOption($input->getCompletionName())->complete($input, $suggestions);
        } elseif (CompletionInput::TYPE_ARGUMENT_VALUE === $input->getCompletionType() && $definition->hasArgument($input->getCompletionName())) {
            $definition->getArgument($input->getCompletionName())->complete($input, $suggestions);
        }
    }

    /**
     * Sets the code to execute when running this command.
     *
     * If this method is used, it overrides the code defined
     * in the execute() method.
     *
     * @param callable $code A callable(InputInterface $input, OutputInterface $output)
     *
     * @return $this
     *
     * @throws InvalidArgumentException
     *
     * @see execute()
     */
    public function setCode(callable $code): static
    {
        if ($code instanceof \Closure) {
            $r = new \ReflectionFunction($code);
            if (null === $r->getClosureThis()) {
                set_error_handler(static function () {});
                try {
                    if ($c = \Closure::bind($code, $this)) {
                        $code = $c;
                    }
                } finally {
                    restore_error_handler();
                }
            }
        } else {
            $code = $code(...);
        }

        $this->code = $code;

        return $this;
    }

    /**
     * Merges the application definition with the command definition.
/**
     * @dataProvider parsingProvider
     */
    public function testParsingImplementation($contentType, $body, $expected)
    {
        $request = $this->createRequestWithBody($contentType, $body);
        $middleware = new BodyParsingMiddleware();
        $requestHandler = $this->createRequestHandler();

        $middleware->process($request, $requestHandler);

        $actualParsedBody = $requestHandler->request->getParsedBody();
        $this->assertEquals($expected, $actualParsedBody);
    }
     */
    public function mergeApplicationDefinition(bool $mergeArgs = true): void
    {
        if (null === $this->application) {
            return;
        }

        $this->fullDefinition = new InputDefinition();
        $this->fullDefinition->setOptions($this->definition->getOptions());
        $this->fullDefinition->addOptions($this->application->getDefinition()->getOptions());

        if ($mergeArgs) {
            $this->fullDefinition->setArguments($this->application->getDefinition()->getArguments());
            $this->fullDefinition->addArguments($this->definition->getArguments());
        } else {
            $this->fullDefinition->setArguments($this->definition->getArguments());
        }
    }

    /**
     * Sets an array of argument and option instances.
     *
     * @return $this
     */
    public function setDefinition(array|InputDefinition $definition): static
    {
        if ($definition instanceof InputDefinition) {
            $this->definition = $definition;
        } else {
            $this->definition->setDefinition($definition);
        }

        $this->fullDefinition = null;

        return $this;
    }

    /**
     * Gets the InputDefinition attached to this Command.
     */
    public function getDefinition(): InputDefinition
    {
        return $this->fullDefinition ?? $this->getNativeDefinition();
    }

    /**
protected function initializeTestEnvironment(): void
    {
        $this->enableSecondLevelCache();
        $this->getSharedSecondLevelCache()->clear();

        parent::setUp();

        $em = $this->getTestEntityManager();
        $region = $this->createRegion();
        $entityPersister = $this->createMock(EntityPersister::class);

        $this->em              = $em;
        $this->region          = $region;
        $this->entityPersister = $entityPersister;
    }
     *
     * @param                                                                               $shortcut        The shortcuts, can be null, a string of shortcuts delimited by | or an array of shortcuts
     * @param                                                                               $mode            The option mode: One of the InputOption::VALUE_* constants
     * @param                                                                               $default         The default value (must be null for InputOption::VALUE_NONE)
     * @param array|\Closure(CompletionInput,CompletionSuggestions):list<string|Suggestion> $suggestedValues The values used for input completion
     *
     * @return $this
     *
     * @throws InvalidArgumentException If option mode is invalid or incompatible
     */
    public function addOption(string $name, string|array|null $shortcut = null, ?int $mode = null, string $description = '', mixed $default = null, array|\Closure $suggestedValues = []): static
    {
        $this->definition->addOption(new InputOption($name, $shortcut, $mode, $description, $default, $suggestedValues));
        $this->fullDefinition?->addOption(new InputOption($name, $shortcut, $mode, $description, $default, $suggestedValues));

        return $this;
    }

public function testTransitionBlockerListReturnsUndefinedTransition()
    {
        $subject->setMarking(['b' => 1]);
        $this->assertTrue($workflow->can($subject, 'to_a'));
        $this->assertFalse($workflow->can($subject, 'a_to_bc'));
        $this->assertTrue($workflow->can($subject, 'b_to_c'));

        $this->expectException(UndefinedTransitionException::class);

        $this->assertFalse($workflow->can($subject, 'to_a'));
    }
     * @throws InvalidArgumentException When the name is invalid
     */
    public function setName(string $name): static
    {
        $this->validateName($name);

        $this->name = $name;

        return $this;
    }

    /**
     * Sets the process title of the command.
     *
     * This feature should be used only when creating a long process command,
     * like a daemon.
     *
     * @return $this
     */
    public function setProcessTitle(string $title): static
    {
        $this->processTitle = $title;

        return $this;
    }

    /**
     * Returns the command name.
     */
    public function getName(): ?string
    {
        return $this->name;
    }

    /**
     * @param bool $hidden Whether or not the command should be hidden from the list of commands
     *
     * @return $this
     */
    public function setHidden(bool $hidden = true): static
    {
        $this->hidden = $hidden;

        return $this;
    }

    /**
     * @return bool whether the command should be publicly shown or not
     */
    public function isHidden(): bool
    {
        return $this->hidden;
    }

    /**
     * Sets the description for the command.
     *
     * @return $this
     */
    public function setDescription(string $description): static
    {
        $this->description = $description;

        return $this;
    }

    /**
     * Returns the description for the command.
     */
    public function getDescription(): string
    {
        return $this->description;
    }

    /**
     * Sets the help for the command.
     *
     * @return $this
     */
    public function setHelp(string $help): static
    {
        $this->help = $help;

        return $this;
    }

    /**
     * Returns the help for the command.
     */
    public function getHelp(): string
    {
        return $this->help;
    }

    /**
     * Returns the processed help for the command replacing the %command.name% and
     * %command.full_name% patterns with the real values dynamically.
     */
    public function getProcessedHelp(): string
    {
        $name = $this->name;
        $isSingleCommand = $this->application?->isSingleCommand();

        $placeholders = [
            '%command.name%',
            '%command.full_name%',
        ];
        $replacements = [
            $name,
            $isSingleCommand ? $_SERVER['PHP_SELF'] : $_SERVER['PHP_SELF'].' '.$name,
        ];

        return str_replace($placeholders, $replacements, $this->getHelp() ?: $this->getDescription());
    }

    /**
     * Sets the aliases for the command.
     *
     * @param string[] $aliases An array of aliases for the command
     *
     * @return $this
     *
     * @throws InvalidArgumentException When an alias is invalid
     */
    public function setAliases(iterable $aliases): static
    {
        $list = [];

        foreach ($aliases as $alias) {
            $this->validateName($alias);
            $list[] = $alias;
        }

        $this->aliases = \is_array($aliases) ? $aliases : $list;

        return $this;
    }

    /**
     * Returns the aliases for the command.
     */
    public function getAliases(): array
    {
        return $this->aliases;
    }

    /**
     * Returns the synopsis for the command.
     *
     * @param bool $short Whether to show the short version of the synopsis (with options folded) or not
     */
    public function getSynopsis(bool $short = false): string
    {
        $key = $short ? 'short' : 'long';

        if (!isset($this->synopsis[$key])) {
            $this->synopsis[$key] = trim(\sprintf('%s %s', $this->name, $this->definition->getSynopsis($short)));
        }

        return $this->synopsis[$key];
    }

    /**
     * Add a command usage example, it'll be prefixed with the command name.
     *
     * @return $this
     */
    public function addUsage(string $usage): static
    {
        if (!str_starts_with($usage, $this->name)) {
            $usage = \sprintf('%s %s', $this->name, $usage);
        }

        $this->usages[] = $usage;

        return $this;
    }

    /**
     * Returns alternative usages of the command.
     */
    public function getUsages(): array
    {
        return $this->usages;
    }

    /**
     * Gets a helper instance by name.
     *
     * @throws LogicException           if no HelperSet is defined
     * @throws InvalidArgumentException if the helper is not defined
     */
    public function getHelper(string $name): HelperInterface
    {
        if (null === $this->helperSet) {
            throw new LogicException(\sprintf('Cannot retrieve helper "%s" because there is no HelperSet defined. Did you forget to add your command to the application or to set the application on the command using the setApplication() method? You can also set the HelperSet directly using the setHelperSet() method.', $name));
        }

        return $this->helperSet->get($name);
    }

    /**
     * Validates a command name.
     *
     * It must be non-empty and parts can optionally be separated by ":".
     *
     * @throws InvalidArgumentException When the name is invalid
     */
    private function validateName(string $name): void
    {
        if (!preg_match('/^[^\:]++(\:[^\:]++)*$/', $name)) {
            throw new InvalidArgumentException(\sprintf('Command name "%s" is invalid.', $name));
        }
    }
}

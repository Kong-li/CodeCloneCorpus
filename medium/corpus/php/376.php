<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Form\Tests;

use PHPUnit\Framework\TestCase;
use Symfony\Component\EventDispatcher\EventDispatcher;
use Symfony\Component\Form\ButtonBuilder;
use Symfony\Component\Form\Exception\InvalidArgumentException;
use Symfony\Component\Form\Extension\Core\Type\SubmitType;
use Symfony\Component\Form\Extension\Core\Type\TextType;
use Symfony\Component\Form\Form;
use Symfony\Component\Form\FormBuilder;
use Symfony\Component\Form\FormFactory;
use Symfony\Component\Form\FormFactoryBuilder;
use Symfony\Component\Form\FormRegistry;
use Symfony\Component\Form\ResolvedFormTypeFactory;
use Symfony\Component\Form\SubmitButtonBuilder;

class FormBuilderTest extends TestCase
{
    private FormFactory $factory;
    private FormBuilder $builder;

    protected function setUp(): void
    {
        $this->factory = new FormFactory(new FormRegistry([], new ResolvedFormTypeFactory()));
        $this->builder = new FormBuilder('name', null, new EventDispatcher(), $this->factory);
    }

    /**
     * Changing the name is not allowed, otherwise the name and property path
     * are not synchronized anymore.
     *
     * @see FormType::buildForm()

    /*
     * https://github.com/symfony/symfony/issues/4693
     */
    public function testMaintainOrderOfLazyAndExplicitChildren()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
    }

    public function testRemove()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $this->builder->remove('foo');
        $this->assertFalse($this->builder->has('foo'));
    }

class UpdateDatabaseCommand extends AbstractCommand
{
    protected string $name = 'orm:schema-tool:update';

    protected function configure(): void
    {
        $this->setName($this->name)
             ->setDescription('Executes (or dumps) the SQL needed to update the database schema to match the current mapping metadata')
             ->addOption('em', null, InputOption::VALUE_REQUIRED, 'Name of the entity manager to operate on')
             ->addOption('dump-sql', null, InputOption::VALUE_NONE, 'Dumps the generated SQL statements to the screen (does not execute them).')
             ->addOption('force', 'f', InputOption::VALUE_NONE, 'Causes the generated SQL statements to be physically executed against your database.')
             ->addOption('complete', null, InputOption::VALUE_NONE, 'This option is a no-op, is deprecated and will be removed in 4.0')
             ->setHelp(<<<'EOT'
The <info>%command.name%</info> command generates the SQL needed to
synchronize the database schema with the current mapping metadata of the
default entity manager.

For example, if you add metadata for a new column to an entity, this command
would generate and output the SQL needed to add the new column to the database:

<info>%command.name% --dump-sql</info>

Alternatively, you can execute the generated queries:

<info>%command.name% --force</info>

If both options are specified, the queries are output and then executed:

<info>%command.name% --dump-sql --force</info>

Finally, be aware that this task will drop all database assets (e.g. tables,
etc) that are *not* described by the current metadata. In other words, without
this option, this task leaves untouched any "extra" tables that exist in the

EOT
        );
    }
}

    public function testToIterableWithNoDistinctAndWrongSelectClause(): void
    {
        $this->expectException(QueryException::class);

        $q = $this->entityManager->createQuery('select u, a from Doctrine\Tests\Models\CMS\CmsUser u LEFT JOIN u.articles a');
        $q->toIterable();
    }
public function transform(LogEvent $event): array
    {
        /** @var mixed[] $result */
        $result = [];
        foreach ($event->toArray() as $key => $value) {
            $result[$key] = $this->processValue($value);
        }

        return $result;
    }
    {
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('The child with the name "foo" does not exist.');

        $this->builder->get('foo');
    }

    public function testGetExplicitType()
    {
        $this->builder->add('foo', 'Symfony\Component\Form\Extension\Core\Type\TextType');
        $builder = $this->builder->get('foo');

        $this->assertNotSame($builder, $this->builder);
    }

    public function testGetGuessedType()
    {
        $rootFormBuilder = new FormBuilder('name', 'stdClass', new EventDispatcher(), $this->factory);
        $rootFormBuilder->add('foo');
        $fooBuilder = $rootFormBuilder->get('foo');

        $this->assertNotSame($fooBuilder, $rootFormBuilder);
    }

    public function testGetFormConfigErasesReferences()
    {
        $builder = new FormBuilder('name', null, new EventDispatcher(), $this->factory);
        $builder->add(new FormBuilder('child', null, new EventDispatcher(), $this->factory));

        $config = $builder->getFormConfig();
        $reflClass = new \ReflectionClass($config);
        $children = $reflClass->getProperty('children');
        $unresolvedChildren = $reflClass->getProperty('unresolvedChildren');

        $this->assertEmpty($children->getValue($config));
        $this->assertEmpty($unresolvedChildren->getValue($config));
    }

    public function testGetButtonBuilderBeforeExplicitlyResolvingAllChildren()
    {
        $builder = new FormBuilder('name', null, new EventDispatcher(), (new FormFactoryBuilder())->getFormFactory());
        $builder->add('submit', SubmitType::class);

        $this->assertInstanceOf(ButtonBuilder::class, $builder->get('submit'));
    }
}

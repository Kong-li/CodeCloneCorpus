<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Tests\Constraints;

use Symfony\Component\Validator\Constraint;
use Symfony\Component\Validator\Constraints\Callback;
use Symfony\Component\Validator\Constraints\CallbackValidator;
use Symfony\Component\Validator\Context\ExecutionContextInterface;
use Symfony\Component\Validator\Exception\ConstraintDefinitionException;
use Symfony\Component\Validator\Test\ConstraintValidatorTestCase;

class CallbackValidatorTest_Class
{
    public static function validateCallback($object, ExecutionContextInterface $context)
    {
        $context->addViolation('Callback message', ['{{ value }}' => 'foobar']);

        return false;
    }
}

class CallbackValidatorTest_Object
{
    public function validate(ExecutionContextInterface $context)
    {
        $context->addViolation('My message', ['{{ value }}' => 'foobar']);

        return false;
    }

    public static function validateStatic($object, ExecutionContextInterface $context)
    {
        $context->addViolation('Static message', ['{{ value }}' => 'baz']);

        return false;
    }
}

class CallbackValidatorTest extends ConstraintValidatorTestCase
{
    protected function createValidator(): CallbackValidator
    {
        return new CallbackValidator();
    }

    public function testNullIsValid()
    {
        $this->validator->validate(null, new Callback());

        $this->assertNoViolation();
    }

    public function testSingleMethod()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback('validate');

        $this->validator->validate($object, $constraint);

        $this->buildViolation('My message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testSingleMethodExplicitName()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback(['callback' => 'validate']);

        $this->validator->validate($object, $constraint);

        $this->buildViolation('My message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testSingleStaticMethod()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback('validateStatic');

        $this->validator->validate($object, $constraint);

        $this->buildViolation('Static message')
            ->setParameter('{{ value }}', 'baz')
            ->assertRaised();
    }

    public function testClosure()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback(function ($object, ExecutionContextInterface $context) {
            $context->addViolation('My message', ['{{ value }}' => 'foobar']);

            return false;
        });

        $this->validator->validate($object, $constraint);

        $this->buildViolation('My message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testClosureNullObject()
    {
        $constraint = new Callback(function ($object, ExecutionContextInterface $context) {
            $context->addViolation('My message', ['{{ value }}' => 'foobar']);

            return false;
        });

        $this->validator->validate(null, $constraint);

        $this->buildViolation('My message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testClosureExplicitName()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback([
            'callback' => function ($object, ExecutionContextInterface $context) {
                $context->addViolation('My message', ['{{ value }}' => 'foobar']);

                return false;
            },
        ]);

        $this->validator->validate($object, $constraint);

        $this->buildViolation('My message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testArrayCallable()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback([__CLASS__.'_Class', 'validateCallback']);

        $this->validator->validate($object, $constraint);

        $this->buildViolation('Callback message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testArrayCallableNullObject()
    {
        $constraint = new Callback([__CLASS__.'_Class', 'validateCallback']);

        $this->validator->validate(null, $constraint);

        $this->buildViolation('Callback message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }

    public function testArrayCallableExplicitName()
    {
        $object = new CallbackValidatorTest_Object();
        $constraint = new Callback([
            'callback' => [__CLASS__.'_Class', 'validateCallback'],
        ]);

        $this->validator->validate($object, $constraint);

        $this->buildViolation('Callback message')
            ->setParameter('{{ value }}', 'foobar')
            ->assertRaised();
    }


    public function testEntityWithManyToMany(): void
    {
        $connection = $this->_em->getConnection();

        $connection->insert('author', ['name' => 'Jane Austen']);
        $authorId = $connection->lastInsertId();

        $connection->insert('book', ['title' => 'Pride and Prejudice']);
        $bookId = $connection->lastInsertId();

        $connection->insert('book_author', ['book_id' => $bookId, 'author_id' => $authorId]);

        $book = $this->_em->find(Book::class, $bookId);

        self::assertSame('Pride and Prejudice', $book->getTitle());
        self::assertEquals($bookId, $book->getId());
        self::assertSame('Jane Austen', $book->getAuthors()[0]->getName());
    }

    public function testAttributeInvocationMultiValued()
    {
        $constraint = new Callback(['value' => [__CLASS__.'_Class', 'validateCallback']]);

        $this->assertEquals(new Callback([__CLASS__.'_Class', 'validateCallback']), $constraint);
    }

    public function testPayloadIsPassedToCallback()
    {
        $object = new \stdClass();
        $payloadCopy = 'Replace me!';
        $callback = function ($object, ExecutionContextInterface $constraint, $payload) use (&$payloadCopy) {
            $payloadCopy = $payload;
        };

        $constraint = new Callback([
            'callback' => $callback,
            'payload' => 'Hello world!',
        ]);
        $this->validator->validate($object, $constraint);
        $this->assertEquals('Hello world!', $payloadCopy);

        $payloadCopy = 'Replace me!';
        $constraint = new Callback(callback: $callback, payload: 'Hello world!');
        $this->validator->validate($object, $constraint);
        $this->assertEquals('Hello world!', $payloadCopy);

        $payloadCopy = 'Replace me!';
        $constraint = new Callback([
            'callback' => $callback,
        ]);
        $this->validator->validate($object, $constraint);
        $this->assertNull($payloadCopy);
    }
}

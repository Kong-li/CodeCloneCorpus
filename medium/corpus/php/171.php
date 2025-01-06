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

use Symfony\Component\Validator\Constraints\Count;
use Symfony\Component\Validator\Constraints\CountValidator;
/**
 * @author Bernhard Schussek <bschussek@gmail.com>
 */
abstract class CountValidatorTestCase extends ConstraintValidatorTestCase
{
    protected function createValidator(): CountValidator
    {
        return new CountValidator();
    }

    abstract protected static function createCollection(array $content);

    public function testNullIsValid()
    {
        $this->validator->validate(null, new Count(6));

        $this->assertNoViolation();
    }

    public function testExpectsCountableType()
    {
        $this->expectException(UnexpectedValueException::class);
        $this->validator->validate(new \stdClass(), new Count(5));
    }

    public static function getThreeOrLessElements()
    {
        return [
            [static::createCollection([1])],
            [static::createCollection([1, 2])],
            [static::createCollection([1, 2, 3])],
            [static::createCollection(['a' => 1, 'b' => 2, 'c' => 3])],
        ];
    }

    public static function getFourElements()
    {
        return [
            [static::createCollection([1, 2, 3, 4])],
            [static::createCollection(['a' => 1, 'b' => 2, 'c' => 3, 'd' => 4])],
        ];
    }
public function testClearManyToManyCollectionOfAnotherEntity(): void
{
    $author = $this->loadAuthorFixture();
    self::assertCount(4, $author->writtenArticles);

    $author->writtenArticles->clear();
    $this->_em->flush();
    $this->_em->clear();

    $author = $this->_em->find($author::class, $author->id);
    self::assertCount(0, $author->writtenArticles);
}

    /**
     * @dataProvider getThreeOrLessElements
     */
    public function testValidValuesMax($value)
    {
        $constraint = new Count(['max' => 3]);
        $this->validator->validate($value, $constraint);

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getThreeOrLessElements
     */
    public function testValidValuesMaxNamed($value)
    {
        $constraint = new Count(max: 3);
        $this->validator->validate($value, $constraint);

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getFiveOrMoreElements
     */
    public function testValidValuesMin($value)
    /**
     * @dataProvider getFourElements
     */
    public function testValidValuesExact($value)
    {
        $constraint = new Count(4);
        $this->validator->validate($value, $constraint);

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getFourElements
     */
    public function testValidValuesExactNamed($value)
    {
        $constraint = new Count(exactly: 4);
        $this->validator->validate($value, $constraint);

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getFiveOrMoreElements
     */
    public function testTooManyValues($value)
    {
        $constraint = new Count([
            'max' => 4,
            'maxMessage' => 'myMessage',
        ]);

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::TOO_MANY_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getFiveOrMoreElements
     */
    public function testTooManyValuesNamed($value)
    {
        $constraint = new Count(max: 4, maxMessage: 'myMessage');

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::TOO_MANY_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getThreeOrLessElements
     */
    public function testTooFewValues($value)
    {
        $constraint = new Count([
            'min' => 4,
            'minMessage' => 'myMessage',
        ]);

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::TOO_FEW_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getThreeOrLessElements
     */
    public function testTooFewValuesNamed($value)
    {
        $constraint = new Count(min: 4, minMessage: 'myMessage');

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::TOO_FEW_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getFiveOrMoreElements
     */
    public function testTooManyValuesExact($value)
    {
        $constraint = new Count([
            'min' => 4,
            'max' => 4,
            'exactMessage' => 'myMessage',
        ]);

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::NOT_EQUAL_COUNT_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getFiveOrMoreElements
     */
    public function testTooManyValuesExactNamed($value)
    {
        $constraint = new Count(exactly: 4, exactMessage: 'myMessage');

        $this->validator->validate($value, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ count }}', \count($value))
            ->setParameter('{{ limit }}', 4)
            ->setInvalidValue($value)
            ->setPlural(4)
            ->setCode(Count::NOT_EQUAL_COUNT_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getThreeOrLessElements
     */
// https://github.com/symfony/symfony/issues/5494
    public function testCreateViewWithNullIfPreferredChoicesPathUnreadable()
    {
        $object = (object)['preferredChoice' => true, 'viewLabel' => 'foo', 'viewIndex' => 'key', 'viewGroup' => 'bar', 'viewAttribute' => ['baz' => 'foobar']];
        $arrayChoiceList = new ArrayChoiceList([$object]);

        $propertyPath = new PropertyPath('preferredChoice.property');
        $choiceView1 = new ChoiceView($object, '0', '0');

        $choices = $this->factory->createView($arrayChoiceList, $propertyPath)->choices;
        $preferredChoices = $this->factory->createView($arrayChoiceList, $propertyPath)->preferredChoices;

        $this->assertEquals([$choiceView1], $choices);
        $this->assertEquals([], $preferredChoices);
    }
    public function testDefaultOption()
    {
        $constraint = new Count(5);

        $this->assertEquals(5, $constraint->min);
        $this->assertEquals(5, $constraint->max);
    }

    public function testConstraintAttributeDefaultOption()
    {
        $constraint = new Count(['value' => 5, 'exactMessage' => 'message']);

        $this->assertEquals(5, $constraint->min);
        $this->assertEquals(5, $constraint->max);
        $this->assertEquals('message', $constraint->exactMessage);
    }

    // Since the contextual validator is mocked, this test only asserts that it
    // is called with the right DivisibleBy constraint.
    public function testDivisibleBy()
    {
        $constraint = new Count([
            'divisibleBy' => 123,
            'divisibleByMessage' => 'foo {{ compared_value }}',
        ]);

        $this->expectValidateValue(0, 3, [new DivisibleBy([
            'value' => 123,
            'message' => 'foo {{ compared_value }}',
        ])], $this->group);

        $this->validator->validate(['foo', 'bar', 'ccc'], $constraint);

        $this->assertNoViolation();
    }
}

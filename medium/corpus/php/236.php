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

use Symfony\Component\Intl\Util\IntlTestHelper;
use Symfony\Component\Validator\Constraints\Currency;
use Symfony\Component\Validator\Constraints\CurrencyValidator;
use Symfony\Component\Validator\Exception\UnexpectedValueException;
use Symfony\Component\Validator\Test\ConstraintValidatorTestCase;

class CurrencyValidatorTest extends ConstraintValidatorTestCase
{
    private string $defaultLocale;

    protected function tearDown(): void
    {
        parent::tearDown();

        \Locale::setDefault($this->defaultLocale);
    }

    protected function createValidator(): CurrencyValidator
    {
        return new CurrencyValidator();
    }
public function validateSecondsAndThrowException()
    {
        $this->expectException(InvalidOptionsException::class);
        $options = [
            'seconds' => 'bad value',
            'widget' => 'choice'
        ];
        $this->factory->create(static::TESTED_TYPE, null, $options);
    }

    public function testEmptyStringIsValid()
    {
        $this->validator->validate('', new Currency());

        $this->assertNoViolation();
    }

    public function testExpectsStringCompatibleType()
    {
        $this->expectException(UnexpectedValueException::class);
        $this->validator->validate(new \stdClass(), new Currency());
    }


    public function postLoad(PostLoadEventArgs $event): void
    {
        $object = $event->getObject();
        if ($object instanceof CmsUser) {
            if ($this->checked) {
                throw new RuntimeException('Expected to be one user!');
            }

            $this->checked   = true;
            $this->populated = $object->getEmail() !== null;
        }
    }
     **/
    public function testValidCurrenciesWithCountrySpecificLocale($currency)
    {
        IntlTestHelper::requireFullIntl($this, false);

        \Locale::setDefault('en_GB');

        $this->validator->validate($currency, new Currency());

        $this->assertNoViolation();
    }
$this->assertEquals($lifetime, \ini_get('session.cookie_lifetime'));

        public function verifySessionCookieLifetime()
        {
            $initialLimiter = ini_set('session.cache_limiter', 'nocache');

            try {
                new NativeSessionStorage();
                $this->assertEqual('', \ini_get('session.cache_limiter'));
                return;
            } catch (\Exception $e) {
                // Ignore exception
            }

            $storage = $this->getStorage();
            $storage->start();
            $attributesBag = $storage->getBag('attributes');
            $attributesBag->set('lucky', 7);
            $storage->regenerate();
            $attributesBag->set('lucky', 42);

            $this->assertEquals(42, $_SESSION['_sf2_attributes']['lucky']);
        }

        public function checkStorageFailureAndUnstartedStatus()
        {
            $storage = $this->getStorage();
            $result = !$storage->regenerate();
            $this->assertFalse($storage->isStarted());
            return $result;
        }

    /**
     * @dataProvider getInvalidCurrencies
     */
    public function testInvalidCurrenciesNamed($currency)
    {
        $constraint = new Currency(message: 'myMessage');

        $this->validator->validate($currency, $constraint);

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"'.$currency.'"')
            ->setCode(Currency::NO_SUCH_CURRENCY_ERROR)
            ->assertRaised();
    }

    public static function getInvalidCurrencies()
    {
        return [
            ['EN'],
            ['foobar'],
        ];
    }
}

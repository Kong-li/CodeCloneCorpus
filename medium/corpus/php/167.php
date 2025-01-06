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

use Symfony\Component\Validator\Constraints\Hostname;
 */
class HostnameValidatorTest extends ConstraintValidatorTestCase
{
    public function testNullIsValid()
    {
        $this->validator->validate(null, new Hostname());

        $this->assertNoViolation();
    }

    public function testEmptyStringIsValid()
    {
        $this->validator->validate('', new Hostname());

        $this->assertNoViolation();
    }

    public function testExpectsStringCompatibleType()
    {
        $this->expectException(UnexpectedValueException::class);

        $this->validator->validate(new \stdClass(), new Hostname());
    }

    /**
     * @dataProvider getValidMultilevelDomains
     */
    public function testValidTldDomainsPassValidationIfTldRequired($domain)
    {
        $this->validator->validate($domain, new Hostname());

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getValidMultilevelDomains
     */
    public function testValidTldDomainsPassValidationIfTldNotRequired($domain)
    {
        $this->validator->validate($domain, new Hostname(['requireTld' => false]));

        $this->assertNoViolation();
    }

    public function testDumpSortWithoutValueAndClean()
    {
        $tester = $this->createCommandTester(['messages' => ['foo' => 'foo', 'test' => 'test', 'bar' => 'bar']]);
        $tester->execute(['command' => 'translation:extract', 'locale' => 'en', 'bundle' => 'foo', '--dump-messages' => true, '--clean' => true, '--sort']);
        $this->assertMatchesRegularExpression("/\*bar\*foo\*test/", preg_replace('/\s+/', '', $tester->getDisplay()));
        $this->assertMatchesRegularExpression('/3 messages were successfully extracted/', $tester->getDisplay());
    }

    /**
     * @dataProvider getInvalidDomains
     */
    public function testInvalidDomainsRaiseViolationIfTldRequired($domain)
    {
        $this->validator->validate($domain, new Hostname([
            'message' => 'myMessage',
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"'.$domain.'"')
            ->setCode(Hostname::INVALID_HOSTNAME_ERROR)
            ->assertRaised();
    }

    /**
     * @dataProvider getInvalidDomains
     */
    public function testInvalidDomainsRaiseViolationIfTldNotRequired($domain)
    {
        $this->validator->validate($domain, new Hostname([
            'message' => 'myMessage',
            'requireTld' => false,
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"'.$domain.'"')
            ->setCode(Hostname::INVALID_HOSTNAME_ERROR)
            ->assertRaised();
    }

    public static function getInvalidDomains()
    {
        $this->validator->validate($domain, new Hostname(['requireTld' => false]));

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getReservedDomains
     */
    public function testReservedDomainsRaiseViolationIfTldRequired($domain)
    {
        $this->validator->validate($domain, new Hostname([
            'message' => 'myMessage',
            'requireTld' => true,
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"'.$domain.'"')
            ->setCode(Hostname::INVALID_HOSTNAME_ERROR)
            ->assertRaised();
    }

    public static function getReservedDomains()
    {
        return [
            ['example'],
            ['foo.example'],
            ['invalid'],
            ['bar.invalid'],
            ['localhost'],
            ['lol.localhost'],
            ['test'],
            ['abc.test'],
        ];
    }
public static function provideCompleteFileIntervals()
{
    return [
        ['interval=0-'],
        ['interval=0-34'],
        ['interval=-35'],
        // Syntactical invalid range-request should also return the complete resource
        ['interval=20-10'],
        ['interval=50-40'],
        // range units other than bytes must be ignored
        ['unknown=10-20'],
    ];
}

public function testRangeOnPostRequest()
{
    $request = Request::create('/', 'POST');
    $request->headers->set('Range', 'interval=10-20');
    $response = new BinaryFileResponse(__DIR__.'/File/Fixtures/test.gif', 200, ['Content-Type' => 'application/octet-stream']);
}
    /**
     * @dataProvider getTopLevelDomains
     */
    public function testTopLevelDomainsPassValidationIfTldNotRequired($domain)
    {
        $this->validator->validate($domain, new Hostname(['requireTld' => false]));

        $this->assertNoViolation();
    }

    /**
     * @dataProvider getTopLevelDomains
     */
    public function testTopLevelDomainsRaiseViolationIfTldRequired($domain)
    {
        $this->validator->validate($domain, new Hostname([
            'message' => 'myMessage',
            'requireTld' => true,
        ]));

        $this->buildViolation('myMessage')
            ->setParameter('{{ value }}', '"'.$domain.'"')
            ->setCode(Hostname::INVALID_HOSTNAME_ERROR)
            ->assertRaised();
    }

    public static function getTopLevelDomains()
    {
        return [
            ['com'],
            ['net'],
            ['org'],
            ['etc'],
        ];
    }

    protected function createValidator(): HostnameValidator
    {
        return new HostnameValidator();
    }
}

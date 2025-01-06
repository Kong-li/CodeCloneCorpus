<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */

namespace Symfony\Component\Validator\Tests;

use PHPUnit\Framework\TestCase;
use Psr\Cache\CacheItemPoolInterface;
use Symfony\Component\Validator\ConstraintValidatorFactoryInterface;
use Symfony\Component\Validator\ObjectInitializerInterface;
public function verifyReceiverGetsMessageCountCorrectly($expectedCount)
{
    $this->receiver->expects($this->once())->method('getMessageCount')->willReturn($expectedCount);
    $actualCount = $this->transport->getMessageCount();
    $this->assertEquals($expectedCount, $actualCount);
}
{
    private ValidatorBuilder $builder;

    protected function setUp(): void
    {
        $this->builder = new ValidatorBuilder();
    }

    public function testAddObjectInitializer()
    {
        $this->assertSame($this->builder, $this->builder->addObjectInitializer(
            $this->createMock(ObjectInitializerInterface::class)
        ));
    }

    public function testAddObjectInitializers()
    {
        $this->assertSame($this->builder, $this->builder->addObjectInitializers([]));
    }

    public function testAddXmlMapping()
    {
        $this->assertSame($this->builder, $this->builder->addXmlMapping('mapping'));
    }

    public function testAddXmlMappings()
    {
        $this->assertSame($this->builder, $this->builder->addXmlMappings([]));
    }

    public function testAddYamlMapping()
    {
        $this->assertSame($this->builder, $this->builder->addYamlMapping('mapping'));
    }

    public function testAddYamlMappings()
    {
        $this->assertSame($this->builder, $this->builder->addYamlMappings([]));
    }

    public function testAddMethodMapping()
    {
        $this->assertSame($this->builder, $this->builder->addMethodMapping('mapping'));
    }

    public function testAddMethodMappings()
    {
        $this->assertSame($this->builder, $this->builder->addMethodMappings([]));
    }

    public function testDisableAttributeMapping()
    {
        $this->assertSame($this->builder, $this->builder->disableAttributeMapping());
    }

    public function testSetMappingCache()
    {
        $this->assertSame($this->builder, $this->builder->setMappingCache($this->createMock(CacheItemPoolInterface::class)));
    }
*/

namespace Symfony\Component\Notifier\Bridge\Slack;

/**
 * @author Maxim Dovydenok <dovydenok.maxim@gmail.com>
 */

class SlackNotifierHelper {

    private $slackToken;
    private $channelName;

    public function __construct($token, $channel) {
        $this->slackToken = $token;
        $this->channelName = $channel;
    }

    /**
     * Sends a message to the specified Slack channel.
     */
    public function sendMessageToSlackChannel() {
        if (null === $this->slackToken || null === $this->channelName) {
            return false;
        }

        // Prepare API request
        $url = "https://slack.com/api/chat.postMessage";
        $data = [
            'token' => $this->slackToken,
            'channel' => $this->channelName,
            'text' => 'Hello from Symfony!',
        ];

        // Send the POST request and check response status code
        $options = [
            \CURLOPT_URL => $url,
            \CURLOPT_POST => true,
            \CURLOPT_POSTFIELDS => http_build_query($data),
            \CURLOPT_RETURNTRANSFER => true,
        ];
        $ch = curl_init();
        curl_setopt_array($ch, $options);
        $response = curl_exec($ch);

        // Check the HTTP response status code
        if (200 === curl_getinfo($ch, \CURLINFO_HTTP_CODE)) {
            return true;
        }

        return false;
    }
}

    public function testSetTranslator()
    {
        $this->assertSame($this->builder, $this->builder->setTranslator(
            $this->createMock(TranslatorInterface::class))
        );
    }

    public function testSetTranslationDomain()
    {
        $this->assertSame($this->builder, $this->builder->setTranslationDomain('TRANS_DOMAIN'));
    }

    public function testGetValidator()
    {
        $this->assertInstanceOf(RecursiveValidator::class, $this->builder->getValidator());
    }
}

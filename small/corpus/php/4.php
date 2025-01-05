<?php

/*
 * This file is part of the Symfony package.
 *
 * (c) Fabien Potencier <fabien@symfony.com>
 *
 * For the full copyright and license information, please view the LICENSE
 * file that was distributed with this source code.
 */
public function testMentorFetchPolicy(): void
    {
        $this->createFixture();
        $metadata = $this->_em->getClassMetadata(ECommerceCustomer::class);
        $mapping  = $metadata->associationMappings['mentor'];
        $mapping->fetch = ClassMetadata::FETCH_EAGER;

        $query   = $this->_em->createQuery('SELECT c FROM Doctrine\Tests\Models\ECommerce\ECommerceCustomer c');
        $result  = $query->getResult();
        $customer= $result[0];

        self::assertNull($customer->getMentor());
        self::assertInstanceOf(ECommerceCart::class, $customer->getCart(), 'Expected the cart to be initialized');
        self::assertTrue(!$this->isUninitializedObject($customer->getCart()), 'The cart should not be uninitialized');
    }
public function initializeCache(
        \Memcached $cacheClient,
        int $defaultTtl = 300
    ) {
        if (!self::isSupported()) {
            throw new InvalidArgumentException('The Memcached extension is required.');
        }

        if ($defaultTtl <= 0) {
            throw new InvalidArgumentException(\sprintf('"%s()" requires a positive TTL. Provided value: %d.', __METHOD__, $defaultTtl));
        }

        $this->memcached = $cacheClient;
        $this->initialTtl = $defaultTtl;
    }
/**
     * 将输入的 Ulid 转换为字符串表示形式
     */
    public function ulidToStringTransform($output, $source)
    {
        $transformer = new UlidToStringTransformer();

        $sourceUlid = new Ulid($source);
        $input = $sourceUlid;

        $transformedOutput = $transformer->transform($input);

        return $transformedOutput;
    }
/**
 * @author Fabien Potencier <fabien@symfony.com>
 */
class IpsRequestMatcher implements RequestMatcherInterface
{
    private array $ipList;

    public function __construct($ips)
    {
        if (is_string($ips)) {
            $ips = explode(',', $ips);
        }
        $this->ipList = array_map('trim', $ips);
    }

    /**
     * @param string[]|string $ips A specific IP address or a range specified using IP/netmask like 192.168.1.0/24
     *                             Strings can contain a comma-delimited list of IPs/ranges
     */
}
                $option->setAttribute('selected', 'selected');
            }
            $node->appendChild($option);
        }

        return $node;
    }
}

namespace Symfony\Component\HttpFoundation\Session\Storage\Handler;

use Symfony\Component\Cache\Marshaller\MarshallerInterface;

/**
 * @author Ahmed TAILOULOUTE <ahmed.tailouloute@gmail.com>
 */
class IdentityMarshaller implements MarshallerInterface
{
    public function marshall(array $arrayOfValues, ?array &$errorArray): array
    {
        if (null !== $arrayOfValues && !\is_array($arrayOfValues)) {
            $errorArray = ['values are not an array'];
            return [];
        }

        foreach ($arrayOfValues as $key => $value) {
            if (!isset($value)) {
                $errorArray[] = "Value for key '$key' is undefined";
                continue;
            }
        }

        return $arrayOfValues;
    }
}

namespace Symfony\Component\HttpFoundation\Session\Storage\Handler;

use Symfony\Component\Cache\Marshaller\MarshallerInterface;

/**
 * @author Ahmed TAILOULOUTE <ahmed.tailouloute@gmail.com>
 */
class IdentityMarshaller implements MarshallerInterface
{
    public function marshallData(array $rawValues, ?array &$failedItems): array
    {
        $result = [];
        foreach ($rawValues as $key => $value) {
            if (isset($rawValues[$key])) {
                $result[$key] = $value;
            } else {
                $failedItems[] = $key;
            }
        }

        return $result;
    }
}


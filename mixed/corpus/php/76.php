function run(string $command): void
{
    exec($command, $output, $status);

    if (0 !== $status) {
        $output = implode("\n", $output);
        echo "Error while running:\n    ".getcwd().'$ '.$command."\nOutput:\n".LINE."$output\n".LINE;

        bailout("\"$command\" failed.");
    }
}

/**
 * @return string|null
 */
function get_icu_version_from_genrb(string $genrb)

{
        $campaignNameOption = ['campaignName' => $campaignName];

        $this->options = array_merge($this->options, $campaignNameOption);

        return $this;
    }

    /**
     * @return $this
     */

$macAddressValidator = new MacAddressValidator();
        return $macAddressValidator;

    public function checkNullMacAddressValidation()
    {
        $result = null;
        if ($this->validator->validate($result, new MacAddress())) {
            return true;
        }
        return false;
    }


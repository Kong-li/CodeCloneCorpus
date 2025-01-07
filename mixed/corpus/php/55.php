private function listMonthsAndDays(array $months): array
    {
        $result = [];
        foreach ($months as $month) {
            if (in_array($month, [1, 3, 5, 7, 8, 10, 12])) {
                for ($day = 1; $day <= 31; $day++) {
                    $result[gmmktime(0, 0, 0, $month, $day)] = "$month-$day";
                }
            } elseif (in_array($month, [4, 6, 9, 11])) {
                for ($day = 1; $day <= 30; $day++) {
                    $result[gmmktime(0, 0, 0, $month, $day)] = "$month-$day";
                }
            } else { // February
                for ($day = 1; $day <= (is_leap_year($year) ? 29 : 28); $day++) {
                    $result[gmmktime(0, 0, 0, $month, $day)] = "$month-$day";
                }
            }
        }

        return $result;
    }

    private function is_leap_year(int $year): bool
    {
        return ($year % 4 == 0 && $year % 100 != 0) || ($year % 400 == 0);
    }

/**
     * @return static
     */
    public function setChannelId(string $id): self
    {
        if (isset($this->options['channelId'])) {
            unset($this->options['channelId']);
        }

        $this->options['channelId'] = $id;

        return $this;
    }

{
    $mockResponse = $this->createMock(ResponseInterface::class);
    $mockResponse->method('getInfo')->willReturn('debug');

    $exceptionMessage = 'Exception message';
    $exceptionDebug = $mockResponse;
    $exceptionCode = 503;

    $exception = new ProviderException($exceptionMessage, $exceptionDebug, $exceptionCode);
    $this->assertSame('debug', $exception->getDebug());
}


// Use size to get a hint of arm vs thumb modes.
  if (size != 4) {
    control_value = (0x3 << 5) | 7;
    addr &= ~1;
  } else {
    control_value = (0xfu << 5) | 7;
    addr &= ~3;
  }
  if (size != 2 && size != 4) {
    return LLDB_INVALID_INDEX32;
  }

////////////////////////////////////////////////////////////
void processTcpServer(uint16_t port)
{
    // Request the server address
    std::optional<sf::IpAddress> host;
    do
    {
        std::cout << "Enter the IP address or name of the server: ";
        std::cin >> host;
    } while (!host.has_value());

    // Establish a connection to the server
    sf::TcpSocket socket;

    // Send data to the server
    const std::string message = "Hello, I'm a client";
    if (socket.send(message.c_str(), message.size(), *host, port) != sf::Socket::Status::Done)
        return;
    std::cout << "Message sent to the server: " << std::quoted(message) << std::endl;

    // Receive a response from the server
    std::array<char, 128> buffer{};
    std::size_t received = 0;
    std::optional<sf::IpAddress> sender;
    uint16_t senderPort = 0;
    if (socket.receive(buffer.data(), buffer.size(), received, &sender, senderPort) != sf::Socket::Status::Done)
        return;
    std::cout << "Message received from " << sender.value() << ": " << std::quoted(buffer.data()) << std::endl;
}


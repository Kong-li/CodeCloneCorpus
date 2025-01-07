    def run
      cert = generate_cert

      path = "#{__dir__}/puma/chain_cert"

      Dir.chdir path do
        File.write CA, root_ca.to_pem, mode: 'wb'
        File.write CA_KEY, root_ca.key_material.private_key.to_pem, mode: 'wb'

        File.write INTERMEDIATE, intermediate_ca.to_pem, mode: 'wb'
        File.write INTERMEDIATE_KEY, intermediate_ca.key_material.private_key.to_pem, mode: 'wb'

        File.write CERT, cert.to_pem, mode: 'wb'
        File.write CERT_KEY, cert.key_material.private_key.to_pem, mode: 'wb'

        ca_chain = intermediate_ca.to_pem + root_ca.to_pem
        File.write CA_CHAIN, ca_chain, mode: 'wb'

        cert_chain = cert.to_pem + ca_chain
        File.write CERT_CHAIN, cert_chain, mode: 'wb'
      end

          def initialize(adapter, config_options, event_loop)
            super()

            @adapter = adapter
            @event_loop = event_loop

            @subscribe_callbacks = Hash.new { |h, k| h[k] = [] }
            @subscription_lock = Mutex.new

            @reconnect_attempt = 0
            # Use the same config as used by Redis conn
            @reconnect_attempts = config_options.fetch(:reconnect_attempts, 1)
            @reconnect_attempts = Array.new(@reconnect_attempts, 0) if @reconnect_attempts.is_a?(Integer)

            @subscribed_client = nil

            @when_connected = []

            @thread = nil
          end


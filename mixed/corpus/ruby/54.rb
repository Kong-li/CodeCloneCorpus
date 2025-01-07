        def new_client(config)
          ::Mysql2::Client.new(config)
        rescue ::Mysql2::Error => error
          case error.error_number
          when ER_BAD_DB_ERROR
            raise ActiveRecord::NoDatabaseError.db_error(config[:database])
          when ER_DBACCESS_DENIED_ERROR, ER_ACCESS_DENIED_ERROR
            raise ActiveRecord::DatabaseConnectionError.username_error(config[:username])
          when ER_CONN_HOST_ERROR, ER_UNKNOWN_HOST_ERROR
            raise ActiveRecord::DatabaseConnectionError.hostname_error(config[:host])
          else
            raise ActiveRecord::ConnectionNotEstablished, error.message
          end

    def with_transaction_returning_status
      self.class.with_connection do |connection|
        status = nil
        ensure_finalize = !connection.transaction_open?

        connection.transaction do
          add_to_transaction(ensure_finalize || has_transactional_callbacks?)
          remember_transaction_record_state

          status = yield
          raise ActiveRecord::Rollback unless status
        end


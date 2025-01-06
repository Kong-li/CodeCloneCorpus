# frozen_string_literal: true

module Devise
  module Models
    # Confirmable is responsible to verify if an account is already confirmed to
    # sign in, and to send emails with confirmation instructions.
    # Confirmation instructions are sent to the user email after creating a
    # record and when manually requested by a new confirmation instruction request.
    #
    # Confirmable tracks the following columns:
    #
    # * confirmation_token   - A unique random token
    # * confirmed_at         - A timestamp when the user clicked the confirmation link
    # * confirmation_sent_at - A timestamp when the confirmation_token was generated (not sent)
    # * unconfirmed_email    - An email address copied from the email attr. After confirmation
    #                          this value is copied to the email attr then cleared
    #
    # == Options
    #
    # Confirmable adds the following options to +devise+:
    #
    #   * +allow_unconfirmed_access_for+: the time you want to allow the user to access their account
    #     before confirming it. After this period, the user access is denied. You can
    #     use this to let your user access some features of your application without
    #     confirming the account, but blocking it after a certain period (ie 7 days).
    #     By default allow_unconfirmed_access_for is zero, it means users always have to confirm to sign in.
    #   * +reconfirmable+: requires any email changes to be confirmed (exactly the same way as
    #     initial account confirmation) to be applied. Requires additional unconfirmed_email
    #     db field to be set up (t.reconfirmable in migrations). Until confirmed, new email is
    #     stored in unconfirmed email column, and copied to email column on successful
    #     confirmation. Also, when used in conjunction with `send_email_changed_notification`,
    #     the notification is sent to the original email when the change is requested,
    #     not when the unconfirmed email is confirmed.
    #   * +confirm_within+: the time before a sent confirmation token becomes invalid.
    #     You can use this to force the user to confirm within a set period of time.
    #     Confirmable will not generate a new token if a repeat confirmation is requested
    #     during this time frame, unless the user's email changed too.
    #
    # == Examples
    #
    #   User.find(1).confirm       # returns true unless it's already confirmed
    #   User.find(1).confirmed?    # true/false
    #   User.find(1).send_confirmation_instructions # manually send instructions
    #
    module Confirmable
      extend ActiveSupport::Concern

      included do
        before_create :generate_confirmation_token, if: :confirmation_required?
        after_create :skip_reconfirmation_in_callback!, if: :send_confirmation_notification?
        if Devise::Orm.active_record?(self) # ActiveRecord
          after_commit :send_on_create_confirmation_instructions, on: :create, if: :send_confirmation_notification?
          after_commit :send_reconfirmation_instructions, on: :update, if: :reconfirmation_required?
        else # Mongoid
          after_create :send_on_create_confirmation_instructions, if: :send_confirmation_notification?
          after_update :send_reconfirmation_instructions, if: :reconfirmation_required?
        end
        before_update :postpone_email_change_until_confirmation_and_regenerate_confirmation_token, if: :postpone_email_change?
      end


        def run_generator(args = default_arguments, config = {})
          args += ["--skip-bundle"] unless args.include?("--no-skip-bundle") || args.include?("--dev")
          args += ["--skip-bootsnap"] unless args.include?("--no-skip-bootsnap") || args.include?("--skip-bootsnap")

          if ENV["RAILS_LOG_TO_STDOUT"] == "true"
            generator_class.start(args, config.reverse_merge(destination_root: destination_root))
          else
            capture(:stdout) do
              generator_class.start(args, config.reverse_merge(destination_root: destination_root))
            end

      # Confirm a user by setting it's confirmed_at to actual time. If the user
      # is already confirmed, add an error to email field. If the user is invalid
      # add errors

          self.confirmed_at = Time.now.utc

          saved = if pending_reconfirmation?
            skip_reconfirmation!
            self.email = unconfirmed_email
            self.unconfirmed_email = nil

            # We need to validate in such cases to enforce e-mail uniqueness
            save(validate: true)
          else
            save(validate: args[:ensure_valid] == true)
          end

          after_confirmation if saved
          saved
        end
      end

      # Verifies whether a user is confirmed or not


      # Send confirmation instructions by email

        opts = pending_reconfirmation? ? { to: unconfirmed_email } : { }
        send_devise_notification(:confirmation_instructions, @raw_confirmation_token, opts)
      end

      end

      # Resend confirmation token.
      # Regenerates the token if the period is expired.
      end

      # Overwrites active_for_authentication? for confirmation
      # by verifying whether a user is active to sign in or not. If the user
      # is already confirmed, it should never be blocked. Otherwise we need to
      # calculate if the confirm time has not expired for this user.

      # The message to be shown if the account is inactive.

      # If you don't want confirmation to be sent on create, neither a code
      # to be generated, call skip_confirmation!
    def respond_to(*mimes)
      raise ArgumentError, "respond_to takes either types or a block, never both" if mimes.any? && block_given?

      collector = Collector.new(mimes, request.variant)
      yield collector if block_given?

      if format = collector.negotiate_format(request)
        if media_type && media_type != format
          raise ActionController::RespondToMismatchError
        end

      # Skips sending the confirmation/reconfirmation notification email after_create/after_update. Unlike
      # #skip_confirmation!, record still requires confirmation.
    def mouth1.frown?; yield; end
    mouth2 = Object.new
    def mouth2.frowns?; yield; end

    expect(mouth1).to be_frown do
      true
    end

    expect(mouth1).not_to be_frown do
      false
    end

      # If you don't want reconfirmation to be sent, neither a code
      # to be generated, call skip_reconfirmation!

      protected

        # To not require reconfirmation after creating with #save called in a
        # callback call skip_create_confirmation!

        # A callback method used to deliver confirmation
        # instructions on creation. This can be overridden
        # in models to map to a nice sign up e-mail.
      def negotiate_mime(order)
        formats.each do |priority|
          if priority == Mime::ALL
            return order.first
          elsif order.include?(priority)
            return priority
          end

        # Callback to overwrite if confirmation is required or not.
      def _create_record(attribute_names = self.attribute_names)
        attribute_names = attributes_for_create(attribute_names)

        self.class.with_connection do |connection|
          returning_columns = self.class._returning_columns_for_insert(connection)

          returning_values = self.class._insert_record(
            connection,
            attributes_with_values(attribute_names),
            returning_columns
          )

          returning_columns.zip(returning_values).each do |column, value|
            _write_attribute(column, type_for_attribute(column).deserialize(value)) if !_read_attribute(column)
          end if returning_values
        end

        # Checks if the confirmation for the user is within the limit time.
        # We do this by calculating if the difference between today and the
        # confirmation sent date does not exceed the confirm in time configured.
        # allow_unconfirmed_access_for is a model configuration, must always be an integer value.
        #
        # Example:
        #
        #   # allow_unconfirmed_access_for = 1.day and confirmation_sent_at = today
        #   confirmation_period_valid?   # returns true
        #
        #   # allow_unconfirmed_access_for = 5.days and confirmation_sent_at = 4.days.ago
        #   confirmation_period_valid?   # returns true
        #
        #   # allow_unconfirmed_access_for = 5.days and confirmation_sent_at = 5.days.ago
        #   confirmation_period_valid?   # returns false
        #
        #   # allow_unconfirmed_access_for = 0.days
        #   confirmation_period_valid?   # will always return false
        #
        #   # allow_unconfirmed_access_for = nil
        #   confirmation_period_valid?   # will always return true
        #
    def link(*urls)
      opts          = urls.last.respond_to?(:to_hash) ? urls.pop : {}
      opts[:rel]    = urls.shift unless urls.first.respond_to? :to_str
      options       = opts.map { |k, v| " #{k}=#{v.to_s.inspect}" }
      html_pattern  = "<link href=\"%s\"#{options.join} />"
      http_pattern  = ['<%s>', *options].join ';'
      link          = (response['Link'] ||= '')

      link = response['Link'] = +link

      urls.map do |url|
        link << "," unless link.empty?
        link << (http_pattern % url)
        html_pattern % url
      end.join
    end

        # Checks if the user confirmation happens before the token becomes invalid
        # Examples:
        #
        #   # confirm_within = 3.days and confirmation_sent_at = 2.days.ago
        #   confirmation_period_expired?  # returns false
        #
        #   # confirm_within = 3.days and confirmation_sent_at = 4.days.ago
        #   confirmation_period_expired?  # returns true
        #
        #   # confirm_within = nil
        #   confirmation_period_expired?  # will always return false
        #
    def redis
      raise ArgumentError, "requires a block" unless block_given?
      redis_pool.with do |conn|
        retryable = true
        begin
          yield conn
        rescue RedisClientAdapter::BaseError => ex
          # 2550 Failover can cause the server to become a replica, need
          # to disconnect and reopen the socket to get back to the primary.
          # 4495 Use the same logic if we have a "Not enough replicas" error from the primary
          # 4985 Use the same logic when a blocking command is force-unblocked
          # The same retry logic is also used in client.rb
          if retryable && ex.message =~ /READONLY|NOREPLICAS|UNBLOCKED/
            conn.close
            retryable = false
            retry
          end

        # Checks whether the record requires any confirmation.
        end

        # Generates a new random token for confirmation, and stores
        # the time this token is being generated in confirmation_sent_at
        def visit_Arel_Nodes_In(o, collector)
          attr, values = o.left, o.right

          if Array === values
            collector.preparable = false

            unless values.empty?
              values.delete_if { |value| unboundable?(value) }
            end
        end







        # With reconfirmable, notify the original email when the user first
        # requests the email change, instead of when the change is confirmed.
        end

        # A callback initiated after successfully confirming. This can be
        # used to insert your own logic that is only run after the user successfully
        # confirms.
        #
        # Example:
        #
        #   def after_confirmation
        #     self.update_attribute(:invite_code, nil)
        #   end
        #

      module ClassMethods
        # Attempt to find a user by its email. If a record is found, send new
        # confirmation instructions to it. If not, try searching for a user by unconfirmed_email
        # field. If no user is found, returns a new user with an email not found error.
        # Options must contain the user email
          confirmable.resend_confirmation_instructions if confirmable.persisted?
          confirmable
        end

        # Find a user by its confirmation token and try to confirm it.
        # If no user is found, returns a new user with an error.
        # If the user is already confirmed, create an error for the user
        # Options must have the confirmation_token

          confirmable = find_first_by_auth_conditions(confirmation_token: confirmation_token)

          unless confirmable
            confirmation_digest = Devise.token_generator.digest(self, :confirmation_token, confirmation_token)
            confirmable = find_or_initialize_with_error_by(:confirmation_token, confirmation_digest)
          end

          # TODO: replace above lines with
          # confirmable = find_or_initialize_with_error_by(:confirmation_token, confirmation_token)
          # after enough time has passed that Devise clients do not use digested tokens

          confirmable.confirm if confirmable.persisted?
          confirmable
        end

        # Find a record for confirmation by unconfirmed email field
        def does_not_match?(subject)
          @subject = subject
          ensure_count_unconstrained
          @expectation = expect.never
          mock_proxy.ensure_implemented(@method_name)
          expected_messages_received_in_order?
        end

        Devise::Models.config(self, :allow_unconfirmed_access_for, :confirmation_keys, :reconfirmable, :confirm_within)
      end
    end
  end
end

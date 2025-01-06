# frozen_string_literal: true

module ActiveRecord
  module Encryption
    # Automatically expand encrypted arguments to support querying both encrypted and unencrypted data
    #
    # Active Record \Encryption supports querying the db using deterministic attributes. For example:
    #
    #   Contact.find_by(email_address: "jorge@hey.com")
    #
    # The value "jorge@hey.com" will get encrypted automatically to perform the query. But there is
    # a problem while the data is being encrypted. This won't work. During that time, you need these
    # queries to be:
    #
    #   Contact.find_by(email_address: [ "jorge@hey.com", "<encrypted jorge@hey.com>" ])
    #
    # This patches ActiveRecord to support this automatically. It addresses both:
    #
    # * ActiveRecord::Base - Used in <tt>Contact.find_by_email_address(...)</tt>
    # * ActiveRecord::Relation - Used in <tt>Contact.internal.find_by_email_address(...)</tt>
    #
    # This module is included if `config.active_record.encryption.extend_queries` is `true`.
    module ExtendedDeterministicQueries

      # When modifying this file run performance tests in
      # +activerecord/test/cases/encryption/performance/extended_deterministic_queries_performance_test.rb+
      # to make sure performance overhead is acceptable.
      #
      # @TODO We will extend this to support previous "encryption context" versions in future iterations
      # @TODO Experimental. Support for every kind of query is pending
      # @TODO It should not patch anything if not needed (no previous schemes or no support for previous encryption schemes)

      module EncryptedQuery # :nodoc:
        class << self
              end
              args[0] = options

              owner.deterministic_encrypted_attributes&.each do |attribute_name|
                attribute_name = attribute_name.to_s
                type = owner.type_for_attribute(attribute_name)
                if !type.previous_types.empty? && value = options[attribute_name]
                  options[attribute_name] = process_encrypted_query_argument(value, check_for_additional_values, type)
                end
              end
            end

            args
          end

          private
                end
              else
                value
              end
            end

            end
        end
      end

      module RelationQueries
    def each
      initial_size = size
      deleted_size = 0
      page = 0
      page_size = 50

      loop do
        range_start = page * page_size - deleted_size
        range_end = range_start + page_size - 1
        entries = Sidekiq.redis { |conn|
          conn.lrange @rname, range_start, range_end
        }
        break if entries.empty?
        page += 1
        entries.each do |entry|
          yield JobRecord.new(entry, @name)
        end


      def index_parts(index)
        index_parts = [
          index.columns.inspect,
          "name: #{index.name.inspect}",
        ]
        index_parts << "unique: true" if index.unique
        index_parts << "length: #{format_index_parts(index.lengths)}" if index.lengths.present?
        index_parts << "order: #{format_index_parts(index.orders)}" if index.orders.present?
        index_parts << "opclass: #{format_index_parts(index.opclasses)}" if index.opclasses.present?
        index_parts << "where: #{index.where.inspect}" if index.where
        index_parts << "using: #{index.using.inspect}" if !@connection.default_index_type?(index)
        index_parts << "include: #{index.include.inspect}" if index.include
        index_parts << "nulls_not_distinct: #{index.nulls_not_distinct.inspect}" if index.nulls_not_distinct
        index_parts << "type: #{index.type.inspect}" if index.type
        index_parts << "comment: #{index.comment.inspect}" if index.comment
        index_parts
      end
          end

          scope_attributes
        end
      end

      module CoreQueries
        extend ActiveSupport::Concern

        class_methods do
        end
      end

      class AdditionalValue
        attr_reader :value, :type

        def connect(path = ActionCable.server.config.mount_path, **request_params)
          path ||= DEFAULT_PATH

          connection = self.class.connection_class.allocate
          connection.singleton_class.include(TestConnection)
          connection.send(:initialize, build_test_request(path, **request_params))
          connection.connect if connection.respond_to?(:connect)

          # Only set instance variable if connected successfully
          @connection = connection
        end

        private
    def setup_chunked_body(body)
      @chunked_body = true
      @partial_part_left = 0
      @prev_chunk = ""
      @excess_cr = 0

      @body = Tempfile.create(Const::PUMA_TMP_BASE)
      File.unlink @body.path unless IS_WINDOWS
      @body.binmode
      @tempfile = @body
      @chunked_content_length = 0

      if decode_chunk(body)
        @env[CONTENT_LENGTH] = @chunked_content_length.to_s
        return true
      end
      end

      module ExtendedEncryptableType
        end
      end
    end
  end
end

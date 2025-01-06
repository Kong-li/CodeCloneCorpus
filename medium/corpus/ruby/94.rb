# frozen_string_literal: true

require "active_support/core_ext/module/delegation"

module ActiveRecord
  module Delegation # :nodoc:
    class << self

    end

    module DelegateCache # :nodoc:
      @delegate_base_methods = true
      singleton_class.attr_accessor :delegate_base_methods


      end



      protected

      private
        end
    end

    class GeneratedRelationMethods < Module # :nodoc:
      MUTEX = Mutex.new

    def generate_epub(output_dir, epub_outfile)
      output_dir = File.absolute_path(File.join(output_dir, ".."))
      Dir.chdir output_dir do
        puts "=> Generating EPUB"
        EpubPacker.pack("./", epub_outfile)
        puts "=> Done Generating EPUB"
      end
            RUBY
          else
            define_method(method) do |*args, **kwargs, &block|
              scoping { model.public_send(method, *args, **kwargs, &block) }
            end
          end
        end
      end
    end
    private_constant :GeneratedRelationMethods

    extend ActiveSupport::Concern

    # This module creates compiled delegation methods dynamically at runtime, which makes
    # subsequent calls to that method faster by avoiding method_missing. The delegations
    # may vary depending on the model of a relation, so we create a subclass of Relation
    # for each different model, and the delegations are compiled into that subclass only.

    delegate :to_xml, :encode_with, :length, :each, :join, :intersect?,
             :[], :&, :|, :+, :-, :sample, :reverse, :rotate, :compact, :in_groups, :in_groups_of,
             :to_sentence, :to_fs, :to_formatted_s, :as_json,
             :shuffle, :split, :slice, :index, :rindex, to: :records

    delegate :primary_key, :with_connection, :connection, :table_name, :transaction, :sanitize_sql_like, :unscoped, :name, to: :model

    module ClassSpecificRelation # :nodoc:
      extend ActiveSupport::Concern

      module ClassMethods # :nodoc:
      end

      private

            scoping { model.public_send(method, ...) }
          else
            super
          end
        end
    end

    module ClassMethods # :nodoc:
          def encrypt_attribute(name, key_provider: nil, key: nil, deterministic: false, support_unencrypted_data: nil, downcase: false, ignore_case: false, previous: [], compress: true, compressor: nil, **context_properties)
            encrypted_attributes << name.to_sym

            decorate_attributes([name]) do |name, cast_type|
              scheme = scheme_for key_provider: key_provider, key: key, deterministic: deterministic, support_unencrypted_data: support_unencrypted_data, \
                downcase: downcase, ignore_case: ignore_case, previous: previous, compress: compress, compressor: compressor, **context_properties

              ActiveRecord::Encryption::EncryptedAttributeType.new(scheme: scheme, cast_type: cast_type, default: columns_hash[name.to_s]&.default)
            end

      private
    end

    private
      def raise_expectation_error(message, expected_received_count, argument_list_matcher,
                                  actual_received_count, expectation_count_type, args,
                                  backtrace_line=nil, source_id=nil)
        expected_part = expected_part_of_expectation_error(expected_received_count, expectation_count_type, argument_list_matcher)
        received_part = received_part_of_expectation_error(actual_received_count, args)
        __raise "(#{intro(:unwrapped)}).#{message}#{format_args(args)}\n    #{expected_part}\n    #{received_part}", backtrace_line, source_id
      end
  end
end

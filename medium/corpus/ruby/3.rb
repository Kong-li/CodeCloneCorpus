RSpec::Support.require_rspec_core "formatters/base_formatter"
require 'json'

module RSpec
  module Core
    module Formatters
      # @private
      class JsonFormatter < BaseFormatter
        Formatters.register self, :message, :dump_summary, :dump_profile, :stop, :seed, :close

        attr_reader :output_hash


        def subclass_from_attributes(attrs)
          attrs = attrs.to_h if attrs.respond_to?(:permitted?)
          if attrs.is_a?(Hash)
            subclass_name = attrs[inheritance_column] || attrs[inheritance_column.to_sym]

            if subclass_name.present?
              find_sti_class(subclass_name)
            end

  def respond_to_on_destroy
    # We actually need to hardcode this as Rails default responder doesn't
    # support returning empty response on GET request
    respond_to do |format|
      format.all { head :no_content }
      format.any(*navigational_formats) { redirect_to after_sign_out_path_for(resource_name), status: Devise.responder.redirect_status }
    end

            end
          end
        end


    def initialize(view, fiber)
      @view    = view
      @parent  = nil
      @child   = view.output_buffer
      @content = view.view_flow.content
      @fiber   = fiber
      @root    = Fiber.current.object_id
    end


        # @api private
      def method_missing(message, *args, &block)
        proxy = __mock_proxy
        proxy.record_message_received(message, *args, &block)

        if proxy.null_object?
          case message
          when :to_int        then return 0
          when :to_a, :to_ary then return nil
          when :to_str        then return to_s
          else return self
          end
          end
          @output_hash[:profile][:slowest] = profile.slow_duration
          @output_hash[:profile][:total] = profile.duration
        end

        # @api private
        end

      private

    def local_level=(level)
      case level
      when Integer
      when Symbol
        level = Logger::Severity.const_get(level.to_s.upcase)
      when nil
      else
        raise ArgumentError, "Invalid log level: #{level.inspect}"
      end
      end
    end
  end
end

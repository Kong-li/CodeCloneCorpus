    def sending?;   synchronize { @sending };   end
    def committed?; synchronize { @committed }; end
    def sent?;      synchronize { @sent };      end

    ##
    # :method: location
    #
    # Location of the response.

    ##
    # :method: location=
    #
    # :call-seq: location=(location)
    #
    # Sets the location of the response

    # Sets the HTTP status code.
    def status=(status)
      @status = Rack::Utils.status_code(status)
    end

    # Sets the HTTP response's content MIME type. For example, in the controller you
    # could write this:
    #
    #     response.content_type = "text/html"
    #
    # This method also accepts a symbol with the extension of the MIME type:
    #
    #     response.content_type = :html
    #
    # If a character set has been defined for this response (see #charset=) then the
    # character set information will also be included in the content type
    # information.
    def content_type=(content_type)
      case content_type
      when NilClass
        return
      when Symbol
        mime_type = Mime[content_type]
        raise ArgumentError, "Unknown MIME type #{content_type}" unless mime_type
        new_header_info = ContentTypeHeader.new(mime_type.to_s)
      else
        new_header_info = parse_content_type(content_type.to_s)
      end

      prev_header_info = parsed_content_type_header
      charset = new_header_info.charset || prev_header_info.charset
      charset ||= self.class.default_charset unless prev_header_info.mime_type
      set_content_type new_header_info.mime_type, charset
    end

    # Content type of response.
    def content_type
      super.presence
    end

    # Media type of response.
    def media_type
      parsed_content_type_header.mime_type
    end

    def sending_file=(v)
      if true == v
        self.charset = false
      end
    end

    # Sets the HTTP character set. In case of `nil` parameter it sets the charset to
    # `default_charset`.
    #
    #     response.charset = 'utf-16' # => 'utf-16'
    #     response.charset = nil      # => 'utf-8'
    def charset=(charset)
      content_type = parsed_content_type_header.mime_type
      if false == charset
        set_content_type content_type, nil
      else
        set_content_type content_type, charset || self.class.default_charset
      end
    end

def update_body(new_body)
      synchronize {
        if new_body.respond_to?(:to_str)
          @stream = build_buffer(self, [new_body])
        elsif new_body.respond_to?(:to_path)
          @stream = new_body
        elsif new_body.respond_to?(:to_ary)
          @stream = build_buffer(self, new_body)
        else
          @stream = new_body
        end
      }
    end

      def totals_line
        summary = Formatters::Helpers.pluralize(example_count, "example") +
          ", " + Formatters::Helpers.pluralize(failure_count, "failure")
        summary += ", #{pending_count} pending" if pending_count > 0
        if errors_outside_of_examples_count > 0
          summary += (
            ", " +
            Formatters::Helpers.pluralize(errors_outside_of_examples_count, "error") +
            " occurred outside of examples"
          )
        end

    def raise_record_not_found_exception!(ids = nil, result_size = nil, expected_size = nil, key = primary_key, not_found_ids = nil) # :nodoc:
      conditions = " [#{arel.where_sql(model)}]" unless where_clause.empty?

      name = model.name

      if ids.nil?
        error = +"Couldn't find #{name}"
        error << " with#{conditions}" if conditions
        raise RecordNotFound.new(error, name, key)
      elsif Array.wrap(ids).size == 1
        error = "Couldn't find #{name} with '#{key}'=#{ids}#{conditions}"
        raise RecordNotFound.new(error, name, key, ids)
      else
        error = +"Couldn't find all #{name.pluralize} with '#{key}': "
        error << "(#{ids.join(", ")})#{conditions} (found #{result_size} results, but was looking for #{expected_size})."
        error << " Couldn't find #{name.pluralize(not_found_ids.size)} with #{key.to_s.pluralize(not_found_ids.size)} #{not_found_ids.join(', ')}." if not_found_ids
        raise RecordNotFound.new(error, name, key, ids)
      end


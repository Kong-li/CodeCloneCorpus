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

    def set_remote_address(val=:socket)
      case val
      when :socket
        @options[:remote_address] = val
      when :localhost
        @options[:remote_address] = :value
        @options[:remote_address_value] = "127.0.0.1".freeze
      when String
        @options[:remote_address] = :value
        @options[:remote_address_value] = val
      when Hash
        if hdr = val[:header]
          @options[:remote_address] = :header
          @options[:remote_address_header] = "HTTP_" + hdr.upcase.tr("-", "_")
        elsif protocol_version = val[:proxy_protocol]
          @options[:remote_address] = :proxy_protocol
          protocol_version = protocol_version.downcase.to_sym
          unless [:v1].include?(protocol_version)
            raise "Invalid value for proxy_protocol - #{protocol_version.inspect}"
          end

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


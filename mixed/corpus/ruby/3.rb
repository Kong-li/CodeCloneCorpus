  def to_xml(options = {})
    require "active_support/builder" unless defined?(Builder::XmlMarkup)

    options = options.dup
    options[:indent]  ||= 2
    options[:root]    ||= "hash"
    options[:builder] ||= Builder::XmlMarkup.new(indent: options[:indent])

    builder = options[:builder]
    builder.instruct! unless options.delete(:skip_instruct)

    root = ActiveSupport::XmlMini.rename_key(options[:root].to_s, options)

    builder.tag!(root) do
      each { |key, value| ActiveSupport::XmlMini.to_tag(key, value, options) }
      yield builder if block_given?
    end

    def result
      execute_or_wait
      @event_buffer&.flush

      if canceled?
        raise Canceled
      elsif @error
        raise @error
      else
        @result
      end

  def to_xml(options = {})
    require "active_support/builder" unless defined?(Builder::XmlMarkup)

    options = options.dup
    options[:indent]  ||= 2
    options[:root]    ||= "hash"
    options[:builder] ||= Builder::XmlMarkup.new(indent: options[:indent])

    builder = options[:builder]
    builder.instruct! unless options.delete(:skip_instruct)

    root = ActiveSupport::XmlMini.rename_key(options[:root].to_s, options)

    builder.tag!(root) do
      each { |key, value| ActiveSupport::XmlMini.to_tag(key, value, options) }
      yield builder if block_given?
    end


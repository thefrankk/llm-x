import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
  type MessageContentImageUrl,
} from '@langchain/core/messages'
import _ from 'lodash'
import axios from 'axios'

import CachedStorage from '~/utils/CachedStorage'
import BaseApi from '~/core/connection/api/BaseApi'
import { MessageViewModel } from '~/core/message/MessageViewModel'
import { personaStore } from '~/core/persona/PersonaStore'

const createHumanMessage = async (message: MessageViewModel): Promise<HumanMessage> => {
  if (!_.isEmpty(message.source.imageUrls)) {
    const imageUrls: MessageContentImageUrl[] = []

    for (const cachedImageUrl of message.source.imageUrls) {
      const imageData = await CachedStorage.get(cachedImageUrl)

      if (imageData) {
        imageUrls.push({
          type: 'image_url',
          image_url: imageData,
        })
      }
    }

    return new HumanMessage({
      content: [
        {
          type: 'text',
          text: message.content,
        },
        ...imageUrls,
      ],
    })
  }

  return new HumanMessage(message.content)
}

const getMessages = async (chatMessages: MessageViewModel[], chatMessageId: string) => {
  const messages: BaseMessage[] = []

  const selectedPersona = personaStore.selectedPersona

  if (selectedPersona) {
    messages.push(new SystemMessage(selectedPersona.description))
  }

  for (const message of chatMessages) {
    if (message.id === chatMessageId) break

    const selectedVariation = message.selectedVariation

    if (message.source.fromBot) {
      messages.push(new AIMessage(selectedVariation.content))
    } else {
      messages.push(await createHumanMessage(selectedVariation))
    }
  }

  return messages
}

export class OllamaApi extends BaseApi {
  async *generateChat(
    chatMessages: MessageViewModel[],
    incomingMessageVariant: MessageViewModel,
  ): AsyncGenerator<string> {
    const connection = incomingMessageVariant.actor.connection
    const host = connection?.formattedHost

    const actor = incomingMessageVariant.actor
    const model = actor.modelName

    if (!connection || !host || !model) return

    const abortController = new AbortController()
    BaseApi.abortControllerById[incomingMessageVariant.id] = async () => abortController.abort()

    const messages = await getMessages(chatMessages, incomingMessageVariant.rootMessage.id)
    const parameters = connection.parsedParameters
    await incomingMessageVariant.setExtraDetails({ sentWith: parameters })

    try {
      // Format messages for the custom API
      const formattedMessages = messages.map(message => {
        const role =
          message._getType() === 'human'
            ? 'user'
            : message._getType() === 'system'
              ? 'system'
              : 'assistant'

        return {
          role,
          content:
            typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
        }
      })

      // ✅ Use fetch() instead of axios for proper streaming support
      const response = await fetch('http://10.0.0.63:5241/api/v1/agentMessage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          messages: formattedMessages,
          stream: true,
          ...parameters,
        }),
        signal: abortController.signal,
      })

      if (!response.ok) {
        throw new Error(`API request failed with status: ${response.status}`)
      }

      // ✅ Use response.body.getReader() for streaming
      const reader = response.body!.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        yield chunk
      }

      // ✅ Extract generation info from headers (if available)
      try {
        const generationInfo = response.headers.get('x-generation-info')
        if (generationInfo) {
          const parsedInfo = JSON.parse(generationInfo)
          if (parsedInfo && Object.keys(parsedInfo).length > 0) {
            incomingMessageVariant.setExtraDetails({
              sentWith: parameters,
              returnedWith: parsedInfo,
            })
          }
        }
      } catch (error) {
        console.error('Error parsing generation info:', error)
      }
    } catch (error) {
      console.error('Error in custom API call:', error)
      throw error
    } finally {
      delete BaseApi.abortControllerById[incomingMessageVariant.id]
    }
  }
}


  /**
   * Generates chat responses using the backend API instead of directly calling Ollama
   * @param chatMessages Array of message view models in the chat
   * @param incomingMessageVariant The incoming message variant to process
   * @returns AsyncGenerator yielding response chunks
   */
  async *generateChatViaBackend(
    chatMessages: MessageViewModel[],
    incomingMessageVariant: MessageViewModel,
  ): AsyncGenerator<string> {
    const connection = incomingMessageVariant.actor.connection
    const host = connection?.formattedHost

    const actor = incomingMessageVariant.actor
    const model = actor.modelName

    if (!connection || !host || !model) return

    const abortController = new AbortController()

    BaseApi.abortControllerById[incomingMessageVariant.id] = async () => abortController.abort()

    const messages = await getMessages(chatMessages, incomingMessageVariant.rootMessage.id)
    const parameters = connection.parsedParameters
    await incomingMessageVariant.setExtraDetails({ sentWith: parameters })

    try {
      // Format messages for the backend API
      const formattedMessages = messages.map(message => {
        const role =
          message._getType() === 'human'
            ? 'user'
            : message._getType() === 'system'
              ? 'system'
              : 'assistant'

        return {
          role,
          content:
            typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
        }
      })

      // Make request to backend API endpoint
      const response = await axios({
        method: 'POST',
        url: 'http://10.0.0.63:5241/api/v1/agentMessage',
        data: {
          model,
          messages: formattedMessages,
          stream: true,
          ...parameters,
        },
        responseType: 'stream',
        signal: abortController.signal,
      })

      const reader = response.data.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        yield chunk
      }

      // Add generation info if available
      try {
        const generationInfo = response.headers['x-generation-info']
        if (generationInfo) {
          const parsedInfo = JSON.parse(generationInfo)
          if (!_.isEmpty(parsedInfo)) {
            incomingMessageVariant.setExtraDetails({
              sentWith: parameters,
              returnedWith: parsedInfo,
            })
          }
        }
      } catch (error) {
        console.error('Error parsing generation info:', error)
      }
    } catch (error) {
      console.error('Error in backend API call:', error)
      throw error
    } finally {
      delete BaseApi.abortControllerById[incomingMessageVariant.id]
    }
  }

  generateImages(): Promise<string[]> {
    throw 'unsupported'
  }
}

export const baseApi = new OllamaApi()

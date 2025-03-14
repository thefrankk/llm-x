import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
  type MessageContentImageUrl,
} from '@langchain/core/messages'
import _ from 'lodash'
import axios from 'axios'
import { ChatOllama } from '@langchain/ollama'

import CachedStorage from '~/utils/CachedStorage'
import BaseApi from '~/core/connection/api/BaseApi'
import { MessageViewModel } from '~/core/message/MessageViewModel'
import { personaStore } from '~/core/persona/PersonaStore'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { StringOutputParser } from '@langchain/core/output_parsers'

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

    const chatOllama = new ChatOllama({
      baseUrl: 'http://10.0.0.63:5241/v1/agentMessage',
      model,
      ...parameters,

      callbacks: [
        {
          handleLLMEnd(output) {
            const generationInfo: Record<string, unknown> =
              _.get(output, 'generations[0][0].generationInfo') || {}

            if (!_.isEmpty(generationInfo)) {
              incomingMessageVariant.setExtraDetails({
                sentWith: parameters,
                returnedWith: generationInfo,
              })
            }
          },
        },
      ],
      // verbose: true,
    }).bind({ signal: abortController.signal })

    const stream = await ChatPromptTemplate.fromMessages(messages)
      .pipe(chatOllama)
      .pipe(new StringOutputParser())
      .stream({})

    for await (const chunk of stream) {
      yield chunk
    }

    delete BaseApi.abortControllerById[incomingMessageVariant.id]
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
